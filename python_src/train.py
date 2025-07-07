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

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

import sys
sys.path.append('/content/PAQN/build')

# <<< ИСПРАВЛЕНИЕ: Импортируем правильный класс модели
from .model import PAQN_Network 
from ofc_engine import ReplayBuffer, initialize_evaluator, SolverManager, InferenceRequest, InferenceResult

# --- Настройки ---
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
BATCH_SIZE = 1024
MIN_BUFFER_FILL = BATCH_SIZE * 10
SAVE_INTERVAL_SECONDS = 300
STATS_INTERVAL_SECONDS = 10
MODEL_PATH = "/content/models/paqn_model.pth" # <<< ИСПРАВЛЕНИЕ: Одна модель - один путь

class InferenceWorker(mp.Process):
    def __init__(self, name, task_queue, result_queue, stop_event):
        super().__init__(name=name)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.model = None # <<< ИСПРАВЛЕНИЕ: Одна модель
        self.device = None

    def _initialize(self):
        print(f"[{self.name}] Started.", flush=True)
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # <<< ИСПРАВЛЕНИЕ: Инициализируем и загружаем одну модель
        self.model = PAQN_Network(INPUT_SIZE, ACTION_VECTOR_SIZE).to(self.device)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    def run(self):
        self._initialize()
        while not self.stop_event.is_set():
            try:
                req: InferenceRequest = self.task_queue.get(timeout=1.0)
                
                if req.is_policy_request:
                    if not req.action_vectors:
                        predictions = []
                    else:
                        batch_size = len(req.action_vectors)
                        infoset_tensor = torch.tensor([req.infoset] * batch_size, dtype=torch.float32, device=self.device)
                        actions_tensor = torch.tensor(req.action_vectors, dtype=torch.float32, device=self.device)
                        with torch.inference_mode():
                            # <<< ИСПРАВЛЕНИЕ: Вызываем модель для получения только логитов
                            policy_logits, _ = self.model(infoset_tensor, actions_tensor)
                            predictions = policy_logits.cpu().numpy().flatten().tolist()
                    
                    res = InferenceResult()
                    res.id = req.id
                    res.is_policy_result = True
                    res.predictions = predictions
                    self.result_queue.put(res)

                else: # Value request
                    infoset_tensor = torch.tensor([req.infoset], dtype=torch.float32, device=self.device)
                    with torch.inference_mode():
                        # <<< ИСПРАВЛЕНИЕ: Вызываем модель для получения только value
                        value = self.model(infoset_tensor)
                        prediction = value.item()
                    
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
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    print("Initializing C++ hand evaluator lookup tables...", flush=True)
    initialize_evaluator()
    print("C++ evaluator initialized successfully.", flush=True)

    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    
    # <<< ИСПРАВЛЕНИЕ: Одна модель, один оптимизатор
    model = PAQN_Network(INPUT_SIZE, ACTION_VECTOR_SIZE).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr=POLICY_LR) # Можно использовать один LR или разные для групп параметров
    value_criterion = nn.MSELoss()
    policy_criterion = nn.MSELoss()
    
    policy_buffer = ReplayBuffer(BUFFER_CAPACITY)
    value_buffer = ReplayBuffer(BUFFER_CAPACITY)
    
    request_queue = mp.Queue(maxsize=NUM_CPP_WORKERS * 4)
    result_queue = mp.Queue(maxsize=NUM_CPP_WORKERS * 4)
    stop_event = mp.Event()

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
            time.sleep(0.5)
            
            # <<< ИСПРАВЛЕНИЕ: Обучаем обе "головы" за один проход
            if value_buffer.size() >= MIN_BUFFER_FILL and policy_buffer.size() >= MIN_BUFFER_FILL:
                model.train()
                
                # Сэмплируем для value
                v_batch = value_buffer.sample(BATCH_SIZE)
                # Сэмплируем для policy
                p_batch = policy_buffer.sample(BATCH_SIZE)

                if v_batch and p_batch:
                    v_infosets_np, _, v_targets_np = v_batch
                    p_infosets_np, p_actions_np, p_advantages_np = p_batch

                    v_infosets = torch.from_numpy(v_infosets_np).to(device)
                    v_targets = torch.from_numpy(v_targets_np).to(device)
                    p_infosets = torch.from_numpy(p_infosets_np).to(device)
                    p_actions = torch.from_numpy(p_actions_np).to(device)
                    p_advantages = torch.from_numpy(p_advantages_np).to(device)

                    optimizer.zero_grad()
                    
                    # Forward pass для value
                    pred_values = model(v_infosets)
                    loss_v = value_criterion(pred_values, v_targets)
                    
                    # Forward pass для policy
                    pred_logits, _ = model(p_infosets, p_actions)
                    loss_p = policy_criterion(pred_logits, p_advantages)
                    
                    # Суммарный loss
                    total_loss = loss_v + loss_p
                    total_loss.backward()
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    value_losses.append(loss_v.item())
                    policy_losses.append(loss_p.item())

                model.eval()

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
                print("\n--- Saving model ---", flush=True)
                torch.save(model.state_dict(), MODEL_PATH)
                last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping all workers...", flush=True)
        solver_manager.stop()
        print("C++ workers stopped.")
        
        stop_event.set()
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
        torch.save(model.state_dict(), f"{MODEL_PATH}.final")
        print("Training finished.")

if __name__ == "__main__":
    main()
