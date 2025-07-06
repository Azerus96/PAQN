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

from .model import PolicyNetwork, ValueNetwork
from ofc_engine import DeepMCCFR, ReplayBuffer, InferenceQueue, initialize_evaluator

# --- НАСТРОЙКИ ---
NUM_WORKERS = int(os.cpu_count() or 4) # Снизим для отладки, чтобы лог был чище
NUM_COMPUTATION_THREADS = "8"
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_VECTOR_SIZE = 208
ACTION_LIMIT = 1000
POLICY_LR = 0.0002
VALUE_LR = 0.001
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 4096
SAVE_INTERVAL_SECONDS = 300
POLICY_MODEL_PATH = "paqn_policy_model.pth"
VALUE_MODEL_PATH = "paqn_value_model.pth"

print("Initializing C++ hand evaluator lookup tables... (This may take a minute or two)", flush=True)
initialize_evaluator()
print("C++ evaluator initialized successfully.", flush=True)


class InferenceWorker(threading.Thread):
    def __init__(self, policy_net, value_net, queue, device):
        super().__init__(daemon=True)
        self.policy_net = policy_net
        self.value_net = value_net
        self.queue = queue
        self.device = device
        self.stop_event = threading.Event()

    def run(self):
        tid = threading.get_ident()
        print(f"[Py-Worker TID: {tid}] Started.", flush=True)
        self.policy_net.eval()
        self.value_net.eval()

        while not self.stop_event.is_set():
            try:
                print(f"[Py-Worker TID: {tid}] Calling queue.wait(). GIL will be released.", flush=True)
                self.queue.wait()
                print(f"[Py-Worker TID: {tid}] queue.wait() UNBLOCKED. GIL re-acquired.", flush=True)

                requests = self.queue.pop_all()
                print(f"[Py-Worker TID: {tid}] Popped {len(requests)} requests.", flush=True)
                if not requests:
                    continue

                policy_reqs, value_reqs = [], []
                policy_indices, value_indices = [], []

                for i, r in enumerate(requests):
                    if r.is_policy_request():
                        policy_reqs.append(r)
                        policy_indices.append(i)
                    elif r.is_value_request():
                        value_reqs.append(r)
                        value_indices.append(i)

                if policy_reqs:
                    print(f"[Py-Worker TID: {tid}] Processing {len(policy_reqs)} policy requests.", flush=True)
                    infoset_batch, action_batch, req_indices = [], [], []
                    
                    for req in policy_reqs:
                        data = req.get_policy_data()
                        num_actions = len(data.action_vectors)
                        if num_actions > 0:
                            infoset_batch.extend([data.infoset] * num_actions)
                            action_batch.extend(data.action_vectors)
                        req_indices.append(num_actions)
                    
                    if infoset_batch:
                        infosets_tensor = torch.tensor(infoset_batch, dtype=torch.float32, device=self.device)
                        actions_tensor = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
                        with torch.no_grad():
                            predictions = self.policy_net(infosets_tensor, actions_tensor).cpu().numpy().flatten()
                        print(f"[Py-Worker TID: {tid}] Policy net inference DONE.", flush=True)
                        
                        start_idx = 0
                        for i, req in enumerate(policy_reqs):
                            num_actions = req_indices[i]
                            end_idx = start_idx + num_actions
                            result_list = predictions[start_idx:end_idx].tolist()
                            print(f"[Py-Worker TID: {tid}] Setting policy result for request, num_actions={num_actions}", flush=True)
                            req.set_result(result_list)
                            start_idx = end_idx
                        print(f"[Py-Worker TID: {tid}] All policy results set.", flush=True)
                    else: # Handle cases with no actions
                        for req in policy_reqs: req.set_result([])

                if value_reqs:
                    print(f"[Py-Worker TID: {tid}] Processing {len(value_reqs)} value requests.", flush=True)
                    infoset_batch = [r.get_value_data().infoset for r in value_reqs]
                    infosets_tensor = torch.tensor(infoset_batch, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        predictions = self.value_net(infosets_tensor).cpu().numpy().flatten()
                    print(f"[Py-Worker TID: {tid}] Value net inference DONE.", flush=True)
                    
                    for i, req in enumerate(value_reqs):
                        result_list = [predictions[i]]
                        print(f"[Py-Worker TID: {tid}] Setting value result for request", flush=True)
                        req.set_result(result_list)
                    print(f"[Py-Worker TID: {tid}] All value results set.", flush=True)

            except Exception as e:
                print(f"Error in InferenceWorker: {e}", flush=True)
                traceback.print_exc()
        
        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) stopped.", flush=True)

    def stop(self):
        self.stop_event.set()

def push_to_github(commit_message):
    pass # Отключаем на время отладки

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    policy_net = PolicyNetwork(INPUT_SIZE, ACTION_VECTOR_SIZE).to(device)
    value_net = ValueNetwork(INPUT_SIZE).to(device)

    # ... остальной код загрузки моделей ...

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=POLICY_LR)
    value_optimizer = optim.Adam(value_net.parameters(), lr=VALUE_LR)
    
    value_criterion = nn.MSELoss()
    policy_criterion = nn.MSELoss()

    policy_buffer = ReplayBuffer(BUFFER_CAPACITY)
    value_buffer = ReplayBuffer(BUFFER_CAPACITY)

    inference_queue = InferenceQueue()
    inference_worker = InferenceWorker(policy_net, value_net, inference_queue, device)
    inference_worker.start()

    solvers = [DeepMCCFR(ACTION_LIMIT, policy_buffer, value_buffer, inference_queue) for _ in range(NUM_WORKERS)]
    
    stop_event = threading.Event()
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            def worker_loop(solver):
                while not stop_event.is_set():
                    solver.run_traversal()

            print(f"Submitting {NUM_WORKERS} C++ workers...", flush=True)
            futures = {executor.submit(worker_loop, s) for s in solvers}
            
            last_report_time = time.time()
            last_report_head = 0
            training_cycles = 0

            while True:
                time.sleep(0.1) 
                
                trained_this_loop = False
                if value_buffer.get_count() >= BATCH_SIZE:
                    value_net.train()
                    infosets_np, _, targets_np = value_buffer.sample(BATCH_SIZE)
                    # ... остальной код обучения value_net
                    loss = value_criterion(value_net(torch.from_numpy(infosets_np).to(device)), torch.from_numpy(targets_np).to(device))
                    loss.backward()
                    value_optimizer.step()
                    value_losses.append(loss.item())
                    value_net.eval()
                    trained_this_loop = True
                
                if policy_buffer.get_count() >= BATCH_SIZE:
                    policy_net.train()
                    infosets_np, actions_np, advantages_np = policy_buffer.sample(BATCH_SIZE)
                    # ... остальной код обучения policy_net
                    loss = policy_criterion(policy_net(torch.from_numpy(infosets_np).to(device), torch.from_numpy(actions_np).to(device)), torch.from_numpy(advantages_np).to(device))
                    loss.backward()
                    policy_optimizer.step()
                    policy_losses.append(loss.item())
                    policy_net.eval()
                    trained_this_loop = True

                if trained_this_loop:
                    training_cycles += 1
                
                now = time.time()
                if now - last_report_time > 5.0: # Уменьшил интервал для отладки
                    duration = now - last_report_time
                    current_head = policy_buffer.get_head()
                    samples_generated_interval = current_head - last_report_head
                    samples_per_sec = samples_generated_interval / duration if duration > 0 else 0
                    last_report_head = current_head
                    
                    avg_p_loss = sum(policy_losses) / len(policy_losses) if policy_losses else float('nan')
                    avg_v_loss = sum(value_losses) / len(value_losses) if value_losses else float('nan')
                    
                    print(f"\n--- Stats Update ---", flush=True)
                    print(f"Time: {time.strftime('%H:%M:%S')}", flush=True)
                    print(f"Throughput: {samples_per_sec:.2f} samples/s. Total generated: {current_head:,}", flush=True)
                    print(f"Buffers -> Policy: {policy_buffer.get_count()}/{BUFFER_CAPACITY}, Value: {value_buffer.get_count()}/{BUFFER_CAPACITY}", flush=True)
                    print(f"Avg Losses (last 100) -> Policy: {avg_p_loss:.6f}, Value: {avg_v_loss:.6f}", flush=True)
                    print(f"Training cycles in last interval: {training_cycles}", flush=True)
                    training_cycles = 0
                    last_report_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted.", flush=True)
    finally:
        print("Stopping workers...", flush=True)
        stop_event.set()
        inference_worker.stop()
        inference_worker.join()
        print("All workers stopped.")

if __name__ == "__main__":
    main()
