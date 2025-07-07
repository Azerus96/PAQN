import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import traceback
from collections import deque
import multiprocessing as mp

# <<< ИЗМЕНЕНИЕ: Устанавливаем метод 'spawn' для multiprocessing >>>
# Это САМОЕ ВАЖНОЕ ИЗМЕНЕНИЕ. Оно должно быть выполнено до создания
# любых пулов или процессов в основном блоке скрипта.
if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

# Добавляем путь к скомпилированному модулю
import sys
sys.path.append('/content/PAQN/build')

from .model import PolicyNetwork, ValueNetwork
# <<< ИЗМЕНЕНИЕ: Импортируем новые классы из биндингов
from ofc_engine import ReplayBuffer, initialize_evaluator, SolverManager, InferenceRequest, InferenceResult

# --- Настройки ---
# <<< ИЗМЕНЕНИЕ: Более консервативная и надежная настройка воркеров
TOTAL_CPUS = os.cpu_count() or 1
RESERVED_CPUS = 2 if TOTAL_CPUS > 4 else 1
NUM_INFERENCE_WORKERS = max(1, int(TOTAL_CPUS * 0.25))
NUM_CPP_WORKERS = max(1, TOTAL_CPUS - NUM_INFERENCE_WORKERS - RESERVED_CPUS)

print(f"Configuration: {NUM_CPP_WORKERS} C++ workers, {NUM_INFERENCE_WORKERS} Python inference workers.")

# --- Гиперпараметры ---
INPUT_SIZE = 1486
ACTION_VECTOR_SIZE = 208
ACTION_LIMIT = 1000
POLICY_LR = 0.0002
VALUE_LR = 0.001
BUFFER_CAPACITY = 2_000_000
BATCH_SIZE = 1024 # Уменьшил для стабильности на старте
MIN_BUFFER_FILL = BATCH_SIZE * 10 # Начинаем обучение, когда есть хотя бы 10 батчей
SAVE_INTERVAL_SECONDS = 300
STATS_INTERVAL_SECONDS = 10
POLICY_MODEL_PATH = "/content/models/paqn_policy_model.pth"
VALUE_MODEL_PATH = "/content/models/paqn_value_model.pth"

# <<< ИЗМЕНЕНИЕ: Класс воркера теперь наследуется от mp.Process
class InferenceWorker(mp.Process):
    def __init__(self, name, task_queue, result_queue, stop_event):
        super().__init__(name=name)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.policy_net = None
        self.value_net = None
        self.device = None

    def _initialize(self):
        """Инициализация внутри дочернего процесса."""
        print(f"[{self.name}] Started.", flush=True)
        self.device = torch.device("cpu")
        
        # Настраиваем потоки для torch внутри воркера
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        
        self.policy_net = PolicyNetwork(INPUT_SIZE, ACTION_VECTOR_SIZE).to(self.device)
        self.value_net = ValueNetwork(INPUT_SIZE).to(self.device)
        
        if os.path.exists(POLICY_MODEL_PATH):
            self.policy_net.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=self.device))
        if os.path.exists(VALUE_MODEL_PATH):
            self.value_net.load_state_dict(torch.load(VALUE_MODEL_PATH, map_location=self.device))
            
        self.policy_net.eval()
        self.value_net.eval()

    def run(self):
        self._initialize()
        while not self.stop_event.is_set():
            try:
                # <<< ИЗМЕНЕНИЕ: Получаем C++ структуру InferenceRequest
                req: InferenceRequest = self.task_queue.get(timeout=1.0)
                
                if req.is_policy_request:
                    if not req.action_vectors:
                        predictions = []
                    else:
                        batch_size = len(req.action_vectors)
                        infoset_tensor = torch.tensor([req.infoset] * batch_size, dtype=torch.float32, device=self.device)
                        actions_tensor = torch.tensor(req.action_vectors, dtype=torch.float32, device=self.device)
                        with torch.inference_mode():
                            predictions = self.policy_net(infoset_tensor, actions_tensor).cpu().numpy().flatten().tolist()
                    
                    res = InferenceResult()
                    res.id = req.id
                    res.is_policy_result = True
                    res.predictions = predictions
                    self.result_queue.put(res)

                else: # Value request
                    infoset_tensor = torch.tensor([req.infoset], dtype=torch.float32, device=self.device)
                    with torch.inference_mode():
                        prediction = self.value_net(infoset_tensor).item()
                    
                    res = InferenceResult()
                    res.id = req.id
                    res.is_policy_result = False
                    res.predictions = [prediction]
                    self.result_queue.put(res)

            except mp.queues.Empty:
                continue
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                print(f"---!!! EXCEPTION IN {self.name} !!!---", flush=True)
                traceback.print_exc()
        
        print(f"[{self.name}] Stopped.", flush=True)

def main():
    os.makedirs(os.path.dirname(POLICY_MODEL_PATH), exist_ok=True)
    
    print("Initializing C++ hand evaluator lookup tables...", flush=True)
    initialize_evaluator()
    print("C++ evaluator initialized successfully.", flush=True)

    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    
    # Главные модели для обучения
    policy_net = PolicyNetwork(INPUT_SIZE, ACTION_VECTOR_SIZE).to(device)
    value_net = ValueNetwork(INPUT_SIZE).to(device)
    if os.path.exists(POLICY_MODEL_PATH):
        policy_net.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=device))
    if os.path.exists(VALUE_MODEL_PATH):
        value_net.load_state_dict(torch.load(VALUE_MODEL_PATH, map_location=device))
        
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=POLICY_LR)
    value_optimizer = optim.Adam(value_net.parameters(), lr=VALUE_LR)
    value_criterion = nn.MSELoss()
    policy_criterion = nn.MSELoss()
    
    # Общие ресурсы
    policy_buffer = ReplayBuffer(BUFFER_CAPACITY)
    value_buffer = ReplayBuffer(BUFFER_CAPACITY)
    
    # <<< ИЗМЕНЕНИЕ: Используем multiprocessing.Queue
    request_queue = mp.Queue(maxsize=NUM_CPP_WORKERS * 4)
    result_queue = mp.Queue(maxsize=NUM_CPP_WORKERS * 4)
    stop_event = mp.Event()

    # Запуск воркеров инференса
    inference_workers = []
    for i in range(NUM_INFERENCE_WORKERS):
        worker = InferenceWorker(f"InferenceWorker-{i}", request_queue, result_queue, stop_event)
        worker.start()
        inference_workers.append(worker)

    print(f"Creating C++ SolverManager with {NUM_CPP_WORKERS} workers...", flush=True)
    solver_manager = SolverManager(
        NUM_CPP_WORKERS, ACTION_LIMIT, policy_buffer, value_buffer,
        request_queue, result_queue
    )
    print("C++ workers are running in the background.", flush=True)
    
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)
    last_stats_time = time.time()
    last_save_time = time.time()
    
    try:
        while True:
            time.sleep(0.5) # Главный цикл может спать, работа идет в других процессах
            
            # Обучение Value Network
            if value_buffer.size() >= MIN_BUFFER_FILL:
                value_net.train()
                batch = value_buffer.sample(BATCH_SIZE)
                if batch:
                    infosets_np, _, targets_np = batch
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

            # Обучение Policy Network
            if policy_buffer.size() >= MIN_BUFFER_FILL:
                policy_net.train()
                batch = policy_buffer.sample(BATCH_SIZE)
                if batch:
                    infosets_np, actions_np, advantages_np = batch
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

            now = time.time()
            if now - last_stats_time > STATS_INTERVAL_SECONDS:
                total_generated = policy_buffer.total_generated()
                avg_p_loss = np.mean(policy_losses) if policy_losses else float('nan')
                avg_v_loss = np.mean(value_losses) if value_losses else float('nan')
                
                print("\n" + "="*20 + " STATS UPDATE " + "="*20, flush=True)
                print(f"Time: {time.strftime('%H:%M:%S')}", flush=True)
                print(f"Total Generated: {total_generated:,}", flush=True)
                print(f"Buffer Fill -> Policy: {policy_buffer.size():,}/{BUFFER_CAPACITY:,} ({policy_buffer.size()/BUFFER_CAPACITY:.1%}) "
                      f"| Value: {value_buffer.size():,}/{BUFFER_CAPACITY:,} ({value_buffer.size()/BUFFER_CAPACITY:.1%})", flush=True)
                print(f"Avg Losses (last 100) -> Policy: {avg_p_loss:.6f} | Value: {avg_v_loss:.6f}", flush=True)
                print(f"Request Queue: {request_queue.qsize()} | Result Queue: {result_queue.qsize()}", flush=True)
                print("="*54, flush=True)
                last_stats_time = now

            if now - last_save_time > SAVE_INTERVAL_SECONDS:
                print("\n--- Saving models ---", flush=True)
                torch.save(policy_net.state_dict(), POLICY_MODEL_PATH)
                torch.save(value_net.state_dict(), VALUE_MODEL_PATH)
                last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping all workers...", flush=True)
        solver_manager.stop()
        print("C++ workers stopped.")
        
        stop_event.set()
        # Очищаем очереди, чтобы воркеры вышли из .get()
        while not request_queue.empty():
            try: request_queue.get_nowait()
            except: pass
        while not result_queue.empty():
            try: result_queue.get_nowait()
            except: pass
            
        for worker in inference_workers:
            worker.join(timeout=5)
            if worker.is_alive():
                print(f"Terminating worker {worker.name}...", flush=True)
                worker.terminate()
        print("All Python workers stopped.")
        
        print("Final model saving...", flush=True)
        torch.save(policy_net.state_dict(), f"{POLICY_MODEL_PATH}.final")
        torch.save(value_net.state_dict(), f"{VALUE_MODEL_PATH}.final")
        print("Training finished.")

if __name__ == "__main__":
    main()
