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

# --- ИЗМЕНЕНИЕ: Импортируем обе модели и новые типы из C++ ---
from .model import PolicyNetwork, ValueNetwork
from ofc_engine import DeepMCCFR, ReplayBuffer, InferenceQueue

# --- НАСТРОЙКИ ---
NUM_WORKERS = int(os.cpu_count() or 96)
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

class InferenceWorker(threading.Thread):
    def __init__(self, policy_net, value_net, queue, device):
        super().__init__(daemon=True)
        self.policy_net = policy_net
        self.value_net = value_net
        self.queue = queue
        self.device = device
        self.stop_event = threading.Event()

    def run(self):
        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) started.", flush=True)
        self.policy_net.eval()
        self.value_net.eval()

        while not self.stop_event.is_set():
            try:
                self.queue.wait()
                requests = self.queue.pop_all()
                if not requests:
                    continue

                policy_reqs = []
                value_reqs = []
                for r in requests:
                    if r.get_type() == 0: # 0 for PolicyRequestData
                        policy_reqs.append(r)
                    else: # 1 for ValueRequestData
                        value_reqs.append(r)

                if policy_reqs:
                    infoset_batch, action_batch, indices = [], [], []
                    for req in policy_reqs:
                        data = req.get_policy_data()
                        num_actions = len(data.action_vectors)
                        if num_actions > 0:
                            infoset_batch.extend([data.infoset] * num_actions)
                            action_batch.extend(data.action_vectors)
                        indices.append(num_actions)
                    
                    if infoset_batch:
                        infosets_tensor = torch.tensor(infoset_batch, dtype=torch.float32, device=self.device)
                        actions_tensor = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
                        with torch.no_grad():
                            predictions = self.policy_net(infosets_tensor, actions_tensor).cpu().numpy().flatten()
                        
                        start_idx = 0
                        for i, req in enumerate(policy_reqs):
                            num_actions = indices[i]
                            end_idx = start_idx + num_actions
                            req.set_result(predictions[start_idx:end_idx].tolist())
                            start_idx = end_idx
                    else:
                        for req in policy_reqs: req.set_result([])

                if value_reqs:
                    infoset_batch = [r.get_value_data().infoset for r in value_reqs]
                    infosets_tensor = torch.tensor(infoset_batch, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        predictions = self.value_net(infosets_tensor).cpu().numpy().flatten()
                    
                    for i, req in enumerate(value_reqs):
                        req.set_result([predictions[i]])

            except Exception as e:
                print(f"Error in InferenceWorker: {e}", flush=True)
                traceback.print_exc()
        
        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) stopped.", flush=True)

    def stop(self):
        self.stop_event.set()

def push_to_github(commit_message):
    try:
        print("Pushing progress to GitHub...", flush=True)
        subprocess.run(['git', 'config', '--global', 'user.email', 'bot@example.com'], check=True, capture_output=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Training Bot'], check=True, capture_output=True)
        subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
        status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        
        if status_result.stdout:
            print("Changes detected, creating commit...")
            subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True)
            subprocess.run(['git', 'push'], check=True, capture_output=True)
            print("Progress pushed successfully.", flush=True)
        else:
            print("No changes to commit. Skipping push.", flush=True)
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub: {e}", flush=True)
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
    except Exception as e:
        print(f"An unexpected error occurred during git push: {e}", flush=True)

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    policy_net = PolicyNetwork(INPUT_SIZE, ACTION_VECTOR_SIZE).to(device)
    value_net = ValueNetwork(INPUT_SIZE).to(device)

    if os.path.exists(POLICY_MODEL_PATH):
        print(f"Found policy model at {POLICY_MODEL_PATH}. Loading weights...", flush=True)
        policy_net.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=device))
    if os.path.exists(VALUE_MODEL_PATH):
        print(f"Found value model at {VALUE_MODEL_PATH}. Loading weights...", flush=True)
        value_net.load_state_dict(torch.load(VALUE_MODEL_PATH, map_location=device))

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
    git_thread = None
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            def worker_loop(solver):
                while not stop_event.is_set():
                    solver.run_traversal()

            print(f"Submitting {NUM_WORKERS} long-running C++ worker tasks...", flush=True)
            futures = {executor.submit(worker_loop, s) for s in solvers}
            
            last_save_time = time.time()
            last_report_time = time.time()
            
            # Используем общий счетчик сэмплов из одного из буферов (они должны заполняться примерно одинаково)
            last_report_head = 0

            while True:
                time.sleep(0.05) # Небольшая пауза, чтобы не загружать CPU в Python цикле
                
                # Обучение Value Network
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

                # Обучение Policy Network
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

                now = time.time()
                if now - last_report_time > 10.0:
                    duration = now - last_report_time
                    current_head = policy_buffer.get_head() # Отслеживаем по одному буферу
                    samples_generated_interval = current_head - last_report_head
                    last_report_head = current_head
                    samples_per_sec = samples_generated_interval / duration if duration > 0 else 0
                    
                    avg_p_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0
                    avg_v_loss = sum(value_losses) / len(value_losses) if value_losses else 0
                    
                    print(f"\n--- Stats Update ---", flush=True)
                    print(f"Throughput: {samples_per_sec:.2f} samples/s. Total generated: {current_head:,}", flush=True)
                    print(f"Buffers -> Policy: {policy_buffer.get_count()}/{BUFFER_CAPACITY}, Value: {value_buffer.get_count()}/{BUFFER_CAPACITY}", flush=True)
                    print(f"Avg Policy Loss (last 100): {avg_p_loss:.6f}", flush=True)
                    print(f"Avg Value Loss (last 100): {avg_v_loss:.6f}", flush=True)

                    last_report_time = now

                    if now - last_save_time > SAVE_INTERVAL_SECONDS:
                        if git_thread and git_thread.is_alive():
                            print("Previous Git push is still running. Skipping this save.", flush=True)
                        else:
                            print("\n--- Saving models and pushing to GitHub ---", flush=True)
                            torch.save(policy_net.state_dict(), POLICY_MODEL_PATH)
                            torch.save(value_net.state_dict(), VALUE_MODEL_PATH)
                            
                            commit_message = f"PV-Net Training. Samples: {current_head:,}. P-Loss: {avg_p_loss:.6f}, V-Loss: {avg_v_loss:.6f}"
                            
                            git_thread = threading.Thread(target=push_to_github, args=(commit_message,))
                            git_thread.start()
                            
                            last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping workers...", flush=True)
        stop_event.set()
        inference_worker.stop()
        
        if git_thread and git_thread.is_alive():
            print("Waiting for the final Git push to complete...", flush=True)
            git_thread.join()

        print("\n--- Final Save ---", flush=True)
        torch.save(policy_net.state_dict(), f"{POLICY_MODEL_PATH}.final")
        torch.save(value_net.state_dict(), f"{VALUE_MODEL_PATH}.final")
        print("Final models saved. Exiting.")

if __name__ == "__main__":
    main()
