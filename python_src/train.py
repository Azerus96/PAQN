import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
from collections import deque
import queue  # Используем стандартную потокобезопасную очередь Python

# Импортируем наши компоненты
from .model import PolicyNetwork, ValueNetwork
# Убедитесь, что C++ код скомпилирован с gil_scoped_release на wait()
from ofc_engine import DeepMCCFR, ReplayBuffer, InferenceQueue, initialize_evaluator

# --- Глобальные Настройки ---
# Ограничим C++ воркеры, чтобы не перегружать систему и оставить ядра для инференса
# Для 96-ядерной машины можно поставить 80, оставляя 16 для инференса и системы
NUM_CPP_WORKERS = int(os.cpu_count() - 16) if (os.cpu_count() or 0) > 16 else 4
# Выделим достаточное количество потоков для параллельного инференса
NUM_INFERENCE_WORKERS = 16

# Настроим OMP/MKL для PyTorch, чтобы использовать выделенные ядра
COMPUTATION_THREADS = str(NUM_INFERENCE_WORKERS)
os.environ['OMP_NUM_THREADS'] = COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = COMPUTATION_THREADS
torch.set_num_threads(int(COMPUTATION_THREADS))

print(f"Конфигурация: {NUM_CPP_WORKERS} C++ воркеров, {NUM_INFERENCE_WORKERS} потоков для инференса.")

# --- Гиперпараметры ---
INPUT_SIZE = 1486
ACTION_VECTOR_SIZE = 208
ACTION_LIMIT = 1000
POLICY_LR = 0.0002
VALUE_LR = 0.001
BUFFER_CAPACITY = 2_000_000  # Увеличим, т.к. генерация будет быстрее
BATCH_SIZE = 8192         # Увеличим, чтобы лучше утилизировать CPU
SAVE_INTERVAL_SECONDS = 300
POLICY_MODEL_PATH = "paqn_policy_model.pth"
VALUE_MODEL_PATH = "paqn_value_model.pth"


# --- Инициализация C++ части ---
print("Initializing C++ hand evaluator lookup tables...", flush=True)
initialize_evaluator()
print("C++ evaluator initialized successfully.", flush=True)


class InferenceExecutor(threading.Thread):
    """
    Поток-исполнитель. Его единственная задача - выполнять инференс
    на полученных от Диспетчера данных.
    """
    def __init__(self, policy_net, value_net, device):
        super().__init__(daemon=True)
        self.policy_net = policy_net
        self.value_net = value_net
        self.device = device
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()

    def run(self):
        tid = threading.get_ident()
        print(f"[InferenceExecutor TID: {tid}] Started.", flush=True)
        self.policy_net.eval()
        self.value_net.eval()
        while not self.stop_event.is_set():
            try:
                task_type, requests = self.task_queue.get()
                if task_type is None: # Сигнал остановки
                    break

                if task_type == "POLICY":
                    self.process_policy_batch(requests)
                elif task_type == "VALUE":
                    self.process_value_batch(requests)
                
                self.task_queue.task_done()
            except Exception as e:
                print(f"Error in InferenceExecutor TID {tid}: {e}", flush=True)
                traceback.print_exc()
        print(f"[InferenceExecutor TID: {tid}] Stopped.", flush=True)

    def process_policy_batch(self, requests):
        infoset_batch, action_batch, req_indices = [], [], []
        for req in requests:
            data = req.get_policy_data()
            num_actions = len(data.action_vectors)
            if num_actions > 0:
                infoset_batch.extend([data.infoset] * num_actions)
                action_batch.extend(data.action_vectors)
            req_indices.append(num_actions)
        
        if not infoset_batch:
            for req in requests: req.set_result([])
            return

        infosets_tensor = torch.tensor(infoset_batch, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            predictions = self.policy_net(infosets_tensor, actions_tensor).cpu().numpy().flatten()
        
        start_idx = 0
        for i, req in enumerate(requests):
            num_actions = req_indices[i]
            end_idx = start_idx + num_actions
            req.set_result(predictions[start_idx:end_idx].tolist())
            start_idx = end_idx

    def process_value_batch(self, requests):
        infoset_batch = [r.get_value_data().infoset for r in requests]
        if not infoset_batch:
            return
        
        infosets_tensor = torch.tensor(infoset_batch, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            predictions = self.value_net(infosets_tensor).cpu().numpy().flatten()
        
        for i, req in enumerate(requests):
            req.set_result([predictions[i]])

    def submit_task(self, task_type, requests):
        self.task_queue.put((task_type, requests))

    def stop(self):
        self.stop_event.set()
        self.task_queue.put((None, None)) # Сигнал для выхода из get()


class Dispatcher(threading.Thread):
    """
    Диспетчер. Забирает запросы из C++ и распределяет их по пулу
    исполнителей (InferenceExecutor).
    """
    def __init__(self, cpp_queue, inference_executors):
        super().__init__(daemon=True)
        self.cpp_queue = cpp_queue
        self.inference_executors = inference_executors
        self.stop_event = threading.Event()
        self.num_executors = len(inference_executors)

    def run(self):
        tid = threading.get_ident()
        print(f"[Dispatcher TID: {tid}] Started.", flush=True)
        
        executor_idx = 0
        while not self.stop_event.is_set():
            try:
                self.cpp_queue.wait() # Освобождаем GIL, ждем C++
                requests = self.cpp_queue.pop_all()
                if not requests:
                    continue

                # Разделяем запросы по типу
                policy_reqs, value_reqs = [], []
                for r in requests:
                    if r.is_policy_request():
                        policy_reqs.append(r)
                    elif r.is_value_request():
                        value_reqs.append(r)
                
                # Отправляем задачи в пул исполнителей по кругу
                if policy_reqs:
                    self.inference_executors[executor_idx].submit_task("POLICY", policy_reqs)
                    executor_idx = (executor_idx + 1) % self.num_executors
                
                if value_reqs:
                    self.inference_executors[executor_idx].submit_task("VALUE", value_reqs)
                    executor_idx = (executor_idx + 1) % self.num_executors

            except Exception as e:
                print(f"Error in Dispatcher: {e}", flush=True)
                traceback.print_exc()
        print(f"[Dispatcher TID: {tid}] Stopped.", flush=True)

    def stop(self):
        self.stop_event.set()


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    policy_net = PolicyNetwork(INPUT_SIZE, ACTION_VECTOR_SIZE).to(device)
    value_net = ValueNetwork(INPUT_SIZE).to(device)

    if os.path.exists(POLICY_MODEL_PATH):
        print(f"Loading policy model from {POLICY_MODEL_PATH}...", flush=True)
        policy_net.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=device))
    if os.path.exists(VALUE_MODEL_PATH):
        print(f"Loading value model from {VALUE_MODEL_PATH}...", flush=True)
        value_net.load_state_dict(torch.load(VALUE_MODEL_PATH, map_location=device))

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=POLICY_LR)
    value_optimizer = optim.Adam(value_net.parameters(), lr=VALUE_LR)
    
    value_criterion = nn.MSELoss()
    policy_criterion = nn.MSELoss()

    policy_buffer = ReplayBuffer(BUFFER_CAPACITY)
    value_buffer = ReplayBuffer(BUFFER_CAPACITY)

    cpp_inference_queue = InferenceQueue()
    
    # --- Запуск новой архитектуры инференса ---
    inference_executors = []
    for _ in range(NUM_INFERENCE_WORKERS):
        executor = InferenceExecutor(policy_net, value_net, device)
        executor.start()
        inference_executors.append(executor)

    dispatcher = Dispatcher(cpp_inference_queue, inference_executors)
    dispatcher.start()
    # ----------------------------------------

    solvers = [DeepMCCFR(ACTION_LIMIT, policy_buffer, value_buffer, cpp_inference_queue) for _ in range(NUM_CPP_WORKERS)]
    
    stop_event = threading.Event()
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)

    try:
        with ThreadPoolExecutor(max_workers=NUM_CPP_WORKERS) as executor:
            def worker_loop(solver):
                while not stop_event.is_set():
                    solver.run_traversal()

            print(f"Submitting {NUM_CPP_WORKERS} C++ workers to the pool...", flush=True)
            futures = {executor.submit(worker_loop, s) for s in solvers}
            
            last_report_time = time.time()
            last_report_head = 0
            
            while not stop_event.is_set():
                time.sleep(1.0) # Главный поток может спать дольше, он только обучает и отчитывается

                # --- Обучение Value Network ---
                if value_buffer.get_count() >= BATCH_SIZE:
                    value_net.train()
                    infosets_np, _, targets_np = value_buffer.sample(BATCH_SIZE)
                    infosets = torch.from_numpy(infosets_np).to(device)
                    targets = torch.from_numpy(targets_np).to(device)
                    
                    value_optimizer.zero_grad()
                    predictions = value_net(infosets)
                    loss = value_criterion(predictions, targets)
                    loss.backward()
                    clip_grad_norm_(value_net.parameters(), 1.0)
                    value_optimizer.step()
                    value_losses.append(loss.item())
                    value_net.eval()

                # --- Обучение Policy Network ---
                if policy_buffer.get_count() >= BATCH_SIZE:
                    policy_net.train()
                    infosets_np, actions_np, advantages_np = policy_buffer.sample(BATCH_SIZE)
                    infosets = torch.from_numpy(infosets_np).to(device)
                    actions = torch.from_numpy(actions_np).to(device)
                    advantages = torch.from_numpy(advantages_np).to(device)

                    policy_optimizer.zero_grad()
                    logits = policy_net(infosets, actions)
                    loss = policy_criterion(logits, advantages)
                    loss.backward()
                    clip_grad_norm_(policy_net.parameters(), 1.0)
                    policy_optimizer.step()
                    policy_losses.append(loss.item())
                    policy_net.eval()

                # --- Отчет о прогрессе ---
                now = time.time()
                if now - last_report_time > 10.0:
                    duration = now - last_report_time
                    current_head = policy_buffer.get_head()
                    samples_generated_interval = current_head - last_report_head
                    last_report_head = current_head
                    samples_per_sec = samples_generated_interval / duration if duration > 0 else 0
                    
                    avg_p_loss = np.mean(policy_losses) if policy_losses else float('nan')
                    avg_v_loss = np.mean(value_losses) if value_losses else float('nan')
                    
                    print("\n" + "="*20 + " STATS UPDATE " + "="*20, flush=True)
                    print(f"Time: {time.strftime('%H:%M:%S')}", flush=True)
                    print(f"Throughput: {samples_per_sec:,.2f} samples/sec", flush=True)
                    print(f"Total Generated: {current_head:,}", flush=True)
                    print(f"Buffer Fill -> Policy: {policy_buffer.get_count():,}/{BUFFER_CAPACITY:,} ({policy_buffer.get_count()/BUFFER_CAPACITY:.1%}) "
                          f"| Value: {value_buffer.get_count():,}/{BUFFER_CAPACITY:,} ({value_buffer.get_count()/BUFFER_CAPACITY:.1%})", flush=True)
                    print(f"Avg Losses (last 100) -> Policy: {avg_p_loss:.6f} | Value: {avg_v_loss:.6f}", flush=True)
                    print("="*54, flush=True)
                    
                    last_report_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping all workers...", flush=True)
        stop_event.set()
        
        # Корректно останавливаем все потоки
        dispatcher.stop()
        # Нужно "пнуть" C++ очередь, чтобы диспетчер вышел из wait()
        # Простой способ - создать фиктивный объект и положить в очередь
        # Но проще просто подождать, он выйдет по таймауту или когда придет реальный запрос
        dispatcher.join()
        
        for executor in inference_executors:
            executor.stop()
            executor.join()

        print("All Python workers stopped.")

        # Сохранение моделей
        print("Final model saving...", flush=True)
        torch.save(policy_net.state_dict(), f"{POLICY_MODEL_PATH}.final")
        torch.save(value_net.state_dict(), f"{VALUE_MODEL_PATH}.final")
        print("Training finished.")

if __name__ == "__main__":
    main()
