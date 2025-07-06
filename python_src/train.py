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
import queue

from .model import PolicyNetwork, ValueNetwork
from ofc_engine import DeepMCCFR, ReplayBuffer, initialize_evaluator

# --- Настройки ---
NUM_CPP_WORKERS = int(os.cpu_count() - 16) if (os.cpu_count() or 0) > 20 else 8
NUM_INFERENCE_WORKERS = 16

COMPUTATION_THREADS = str(NUM_INFERENCE_WORKERS)
os.environ['OMP_NUM_THREADS'] = COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = COMPUTATION_THREADS
torch.set_num_threads(int(COMPUTATION_THREADS))

print(f"Configuration: {NUM_CPP_WORKERS} C++ workers, {NUM_INFERENCE_WORKERS} Python inference workers.")

# --- Гиперпараметры ---
INPUT_SIZE = 1486
ACTION_VECTOR_SIZE = 208
ACTION_LIMIT = 1000
POLICY_LR = 0.0002
VALUE_LR = 0.001
BUFFER_CAPACITY = 2_000_000
BATCH_SIZE = 8192
SAVE_INTERVAL_SECONDS = 300
POLICY_MODEL_PATH = "paqn_policy_model.pth"
VALUE_MODEL_PATH = "paqn_value_model.pth"

print("Initializing C++ hand evaluator lookup tables...", flush=True)
initialize_evaluator()
print("C++ evaluator initialized successfully.", flush=True)

inference_task_queue = queue.Queue(maxsize=NUM_CPP_WORKERS * 4)

def policy_inference_callback(traversal_id, infoset, action_vectors, responder_callback):
    inference_task_queue.put(("POLICY", traversal_id, infoset, action_vectors, responder_callback))

def value_inference_callback(traversal_id, infoset, responder_callback):
    inference_task_queue.put(("VALUE", traversal_id, infoset, None, responder_callback))

class InferenceWorker(threading.Thread):
    def __init__(self, worker_id, policy_net, value_net, device, task_queue):
        super().__init__(daemon=True, name=f"InferenceWorker-{worker_id}")
        self.worker_id = worker_id
        self.policy_net = policy_net
        self.value_net = value_net
        self.device = device
        self.task_queue = task_queue
        self.stop_event = threading.Event()

    def run(self):
        print(f"[{self.name}] Started.", flush=True)
        self.policy_net.eval()
        self.value_net.eval()
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get()
                if task is None: break

                task_type, traversal_id, infoset, action_vectors, responder_callback = task
                
                if task_type == "POLICY":
                    if not action_vectors:
                        responder_callback([])
                        self.task_queue.task_done()
                        continue
                    
                    # --- ИСПРАВЛЕНИЕ: ВОЗВРАЩАЕМ ЛОГИКУ СБОРКИ БАТЧА ---
                    batch_size = len(action_vectors)
                    infoset_tensor = torch.tensor([infoset] * batch_size, dtype=torch.float32)
                    actions_tensor = torch.tensor(action_vectors, dtype=torch.float32)
                    # ----------------------------------------------------
                    
                    with torch.inference_mode():
                        predictions = self.policy_net(infoset_tensor, actions_tensor).cpu().numpy().flatten()
                    
                    responder_callback(predictions.tolist())

                elif task_type == "VALUE":
                    # --- ИСПРАВЛЕНИЕ: ВОЗВРАЩАЕМ ЛОГИКУ СБОРКИ БАТЧА ---
                    infoset_tensor = torch.tensor([infoset], dtype=torch.float32)
                    # ----------------------------------------------------

                    with torch.inference_mode():
                        prediction = self.value_net(infoset_tensor).item()
                    
                    responder_callback(prediction)

                self.task_queue.task_done()
            except Exception as e:
                print(f"Error in {self.name}: {e}", flush=True)
                traceback.print_exc()
        
        print(f"[{self.name}] Stopped.", flush=True)
    
    def stop(self):
        self.stop_event.set()
        self.task_queue.put(None)

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

    inference_workers = []
    for i in range(NUM_INFERENCE_WORKERS):
        worker = InferenceWorker(i, policy_net, value_net, device, inference_task_queue)
        worker.start()
        inference_workers.append(worker)

    solvers = [
        DeepMCCFR(ACTION_LIMIT, policy_buffer, value_buffer, policy_inference_callback, value_inference_callback) 
        for _ in range(NUM_CPP_WORKERS)
    ]
    
    stop_event = threading.Event()
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)

    try:
        with ThreadPoolExecutor(max_workers=NUM_CPP_WORKERS) as executor:
            def worker_loop(solver):
                while not stop_event.is_set():
                    try:
                        solver.run_traversal()
                    except Exception as e:
                        print(f"Exception in C++ worker: {e}")
                        traceback.print_exc()

            print(f"Submitting {NUM_CPP_WORKERS} C++ workers to the pool...", flush=True)
            futures = {executor.submit(worker_loop, s) for s in solvers}
            
            last_report_time = time.time()
            last_save_time = time.time()
            last_report_head = 0
            
            while not stop_event.is_set():
                time.sleep(1.0)

                # Обучение
                if value_buffer.get_count() >= BATCH_SIZE:
                    value_net.train()
                    infosets_np, _, targets_np = value_buffer.sample(BATCH_SIZE)
                    infosets, targets = torch.from_numpy(infosets_np), torch.from_numpy(targets_np)
                    value_optimizer.zero_grad()
                    loss = value_criterion(value_net(infosets), targets)
                    loss.backward()
                    clip_grad_norm_(value_net.parameters(), 1.0)
                    value_optimizer.step()
                    value_losses.append(loss.item())
                    value_net.eval()

                if policy_buffer.get_count() >= BATCH_SIZE:
                    policy_net.train()
                    infosets_np, actions_np, advantages_np = policy_buffer.sample(BATCH_SIZE)
                    infosets, actions, advantages = torch.from_numpy(infosets_np), torch.from_numpy(actions_np), torch.from_numpy(advantages_np)
                    policy_optimizer.zero_grad()
                    logits = policy_net(infosets, actions)
                    loss = policy_criterion(logits, advantages)
                    loss.backward()
                    clip_grad_norm_(policy_net.parameters(), 1.0)
                    policy_optimizer.step()
                    policy_losses.append(loss.item())
                    policy_net.eval()
                
                # Отчеты
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
                    print(f"Inference Queue Size: {inference_task_queue.qsize()}", flush=True)
                    print("="*54, flush=True)
                    
                    last_report_time = now

                    if now - last_save_time > SAVE_INTERVAL_SECONDS:
                        print("\n--- Saving models ---", flush=True)
                        torch.save(policy_net.state_dict(), POLICY_MODEL_PATH)
                        torch.save(value_net.state_dict(), VALUE_MODEL_PATH)
                        last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping all workers...", flush=True)
        stop_event.set()
        
        for worker in inference_workers:
            worker.stop()
        for worker in inference_workers:
            worker.join()

        print("All Python workers stopped.")

        print("Final model saving...", flush=True)
        torch.save(policy_net.state_dict(), f"{POLICY_MODEL_PATH}.final")
        torch.save(value_net.state_dict(), f"{VALUE_MODEL_PATH}.final")
        print("Training finished.")

if __name__ == "__main__":
    main()
