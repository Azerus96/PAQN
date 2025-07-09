import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import traceback
from collections import deque
import multiprocessing as mp
import queue
import random
import glob

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

# Исправляем пути, если необходимо
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
build_dir = os.path.join(project_root, 'build')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)


from python_src.model import OFC_CNN_Network
from ofc_engine import ReplayBuffer, initialize_evaluator, SolverManager

# --- Константы ---
NUM_FEATURE_CHANNELS = 16
NUM_SUITS = 4
NUM_RANKS = 13
INFOSET_SIZE = NUM_FEATURE_CHANNELS * NUM_SUITS * NUM_RANKS

# --- Настройки ---
TOTAL_CPUS = os.cpu_count() or 94
RESERVED_CPUS = 2 if TOTAL_CPUS > 4 else 1
NUM_INFERENCE_WORKERS = max(1, int(TOTAL_CPUS * 0.25))
NUM_CPP_WORKERS = max(1, TOTAL_CPUS - NUM_INFERENCE_WORKERS - RESERVED_CPUS)

print(f"Configuration: {NUM_CPP_WORKERS} C++ workers, {NUM_INFERENCE_WORKERS} Python inference workers.")

# --- Гиперпараметры ---
ACTION_LIMIT = 4
LEARNING_RATE = 0.0005
BUFFER_CAPACITY = 2_000_000
BATCH_SIZE = 8192
MIN_BUFFER_FILL_RATIO = 0.05
MIN_BUFFER_FILL_SAMPLES = int(BUFFER_CAPACITY * MIN_BUFFER_FILL_RATIO)
MAX_BATCH_SIZE = 2048
BATCH_TIMEOUT_MS = 2.0

# --- Пути и интервалы ---
STATS_INTERVAL_SECONDS = 15
SAVE_INTERVAL_SECONDS = 60 # Увеличим интервал сохранения
MODEL_DIR = "/content/models"
MODEL_PATH = os.path.join(MODEL_DIR, "paqn_model_latest.pth")
OPPONENT_POOL_DIR = os.path.join(MODEL_DIR, "opponent_pool")
MAX_OPPONENTS_IN_POOL = 20

class InferenceWorker(mp.Process):
    def __init__(self, name, task_queue, result_queue, stop_event):
        super().__init__(name=name)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.latest_model = None
        self.opponent_model = None
        self.device = None
        self.opponent_pool_files = []

    def _initialize(self):
        print(f"[{self.name}] Started.", flush=True)
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        
        self.latest_model = OFC_CNN_Network().to(self.device)
        self.opponent_model = OFC_CNN_Network().to(self.device)
        
        self._load_models()
        
        self.latest_model.eval()
        self.opponent_model.eval()

    def _load_models(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.latest_model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            
            self.opponent_pool_files = glob.glob(os.path.join(OPPONENT_POOL_DIR, "*.pth"))
            if self.opponent_pool_files:
                opponent_path = random.choice(self.opponent_pool_files)
                self.opponent_model.load_state_dict(torch.load(opponent_path, map_location=self.device))
            else:
                self.opponent_model.load_state_dict(self.latest_model.state_dict())
        except Exception as e:
            print(f"[{self.name}] Failed to load models: {e}", flush=True)

    def run(self):
        self._initialize()
        
        STREET_START_IDX = 9
        STREET_END_IDX = 14
        TURN_CHANNEL_IDX = 15

        while not self.stop_event.is_set():
            try:
                batch = []
                start_time = time.time()
                while (len(batch) < MAX_BATCH_SIZE and 
                       (time.time() - start_time) * 1000 < BATCH_TIMEOUT_MS):
                    try:
                        item = self.task_queue.get_nowait()
                        batch.append(item)
                    except queue.Empty:
                        if batch: break
                        time.sleep(0.0005)
                
                if not batch: continue

                policy_requests = {}
                value_requests = {}
                for req in batch:
                    req_id, is_policy, infoset, action_vectors = req
                    if is_policy:
                        policy_requests[req_id] = (infoset, action_vectors)
                    else:
                        value_requests[req_id] = infoset

                with torch.inference_mode():
                    # --- Обработка Value запросов (остается как было, т.к. эффективно) ---
                    if value_requests:
                        infosets = list(value_requests.values())
                        infoset_tensor = torch.tensor(infosets, dtype=torch.float32, device=self.device)
                        infoset_tensor = infoset_tensor.view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS)
                        
                        turn_channel = infoset_tensor[:, TURN_CHANNEL_IDX, 0, 0]
                        is_player_turn = turn_channel > 0.5
                        
                        values = torch.zeros(len(infosets), 1, device=self.device)
                        if torch.any(is_player_turn):
                            values[is_player_turn] = self.latest_model(infoset_tensor[is_player_turn])
                        if torch.any(~is_player_turn):
                            values[~is_player_turn] = self.opponent_model(infoset_tensor[~is_player_turn])

                        for i, req_id in enumerate(value_requests.keys()):
                            self.result_queue[req_id] = (req_id, False, [values[i].item()])

                    # --- ПРАВИЛЬНАЯ БАТЧЕВАЯ ОБРАБОТКА POLICY ЗАПРОСОВ ---
                    if policy_requests:
                        # Разделяем запросы для основной модели и для оппонента
                        latest_model_reqs = {}
                        opponent_model_reqs = {}
                        
                        for req_id, (infoset, action_vectors) in policy_requests.items():
                            if not action_vectors:
                                self.result_queue[req_id] = (req_id, True, [])
                                continue
                            
                            # Определяем, какую модель использовать, по каналу хода
                            turn_channel_val = infoset[TURN_CHANNEL_IDX * NUM_SUITS * NUM_RANKS]
                            if turn_channel_val > 0.5:
                                latest_model_reqs[req_id] = (infoset, action_vectors)
                            else:
                                opponent_model_reqs[req_id] = (infoset, action_vectors)

                        # Обрабатываем каждую группу запросов отдельно
                        for model, reqs in [(self.latest_model, latest_model_reqs), (self.opponent_model, opponent_model_reqs)]:
                            if not reqs:
                                continue

                            all_infosets = []
                            all_actions = []
                            all_streets = []
                            action_counts = []
                            req_ids_ordered = []

                            for req_id, (infoset, action_vectors) in reqs.items():
                                num_actions = len(action_vectors)
                                action_counts.append(num_actions)
                                req_ids_ordered.append(req_id)
                                
                                # Собираем все в один большой список
                                all_infosets.extend([infoset] * num_actions)
                                all_actions.extend(action_vectors)

                            # Создаем большие тензоры
                            infoset_tensor = torch.tensor(all_infosets, dtype=torch.float32, device=self.device)
                            infoset_tensor = infoset_tensor.view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS)
                            actions_tensor = torch.tensor(all_actions, dtype=torch.float32, device=self.device)
                            street_vector = infoset_tensor[:, STREET_START_IDX:STREET_END_IDX, 0, 0]
                            
                            # Один вызов модели для всего батча
                            policy_logits, _ = model(infoset_tensor, actions_tensor, street_vector)
                            
                            # Разбираем результаты обратно по запросам
                            predictions = policy_logits.cpu().numpy().flatten()
                            current_pos = 0
                            for i, req_id in enumerate(req_ids_ordered):
                                count = action_counts[i]
                                result_slice = predictions[current_pos : current_pos + count].tolist()
                                self.result_queue[req_id] = (req_id, True, result_slice)
                                current_pos += count

            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                print(f"---!!! EXCEPTION IN {self.name} !!!---", flush=True)
                traceback.print_exc()
        
        print(f"[{self.name}] Stopped.", flush=True)

def update_opponent_pool(model_version):
    if not os.path.exists(MODEL_PATH): return
    pool_files = sorted(glob.glob(os.path.join(OPPONENT_POOL_DIR, "*.pth")), key=os.path.getmtime)
    while len(pool_files) >= MAX_OPPONENTS_IN_POOL:
        os.remove(pool_files.pop(0))
    new_opponent_path = os.path.join(OPPONENT_POOL_DIR, f"paqn_model_v{model_version}.pth")
    torch.save(torch.load(MODEL_PATH), new_opponent_path)
    print(f"Added model version {model_version} to opponent pool.")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
    
    print("Initializing C++ hand evaluator lookup tables...", flush=True)
    initialize_evaluator()
    print("C++ evaluator initialized successfully.", flush=True)

    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    
    model = OFC_CNN_Network().to(device)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Loaded latest model from:", MODEL_PATH)
        except Exception as e:
            print(f"Could not load model, starting from scratch. Error: {e}")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    policy_buffer = ReplayBuffer(BUFFER_CAPACITY)
    value_buffer = ReplayBuffer(BUFFER_CAPACITY)
    
    request_queue = mp.Queue(maxsize=NUM_CPP_WORKERS * 8) # Можно увеличить
    result_dict = mp.Manager().dict()
    stop_event = mp.Event()

    inference_workers = []
    for i in range(NUM_INFERENCE_WORKERS):
        worker = InferenceWorker(f"InferenceWorker-{i}", request_queue, result_dict, stop_event)
        worker.start()
        inference_workers.append(worker)

    print(f"Creating C++ SolverManager with {NUM_CPP_WORKERS} workers...", flush=True)
    solver_manager = SolverManager(
        NUM_CPP_WORKERS, ACTION_LIMIT, policy_buffer, value_buffer,
        request_queue, result_dict
    )
    print("C++ workers are running in the background.", flush=True)
    
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)
    last_stats_time = time.time()
    last_save_time = time.time()
    model_version = 0
    
    STREET_START_IDX = 9
    STREET_END_IDX = 14

    try:
        while True:
            if value_buffer.size() < MIN_BUFFER_FILL_SAMPLES or policy_buffer.size() < MIN_BUFFER_FILL_SAMPLES:
                print(f"Waiting for buffers to fill... Policy: {policy_buffer.size()}/{MIN_BUFFER_FILL_SAMPLES}, Value: {value_buffer.size()}/{MIN_BUFFER_FILL_SAMPLES}", end='\r')
                time.sleep(1)
                last_stats_time = time.time() # Сбрасываем таймер статистики, пока ждем
                continue

            model.train()
            
            v_batch = value_buffer.sample(BATCH_SIZE)
            if not v_batch: continue
            v_infosets_np, _, v_targets_np = v_batch
            v_infosets = torch.from_numpy(v_infosets_np).view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS).to(device)
            v_targets = torch.from_numpy(v_targets_np).unsqueeze(1).to(device)
            pred_values = model(v_infosets)
            loss_v = criterion(pred_values, v_targets)
            
            p_batch = policy_buffer.sample(BATCH_SIZE)
            if not p_batch: continue
            p_infosets_np, p_actions_np, p_advantages_np = p_batch
            p_infosets = torch.from_numpy(p_infosets_np).view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS).to(device)
            p_actions = torch.from_numpy(p_actions_np).to(device)
            p_advantages = torch.from_numpy(p_advantages_np).unsqueeze(1).to(device)
            
            p_street_vector = p_infosets[:, STREET_START_IDX:STREET_END_IDX, 0, 0]
            pred_logits, _ = model(p_infosets, p_actions, p_street_vector)
            loss_p = criterion(pred_logits, p_advantages)

            optimizer.zero_grad()
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
                print(f"Request Queue: {request_queue.qsize()} | Result Dict: {len(result_dict)}", flush=True)
                print("="*54, flush=True)
                last_stats_time = now

            if now - last_save_time > SAVE_INTERVAL_SECONDS:
                print("\n--- Saving models and updating opponent pool ---", flush=True)
                torch.save(model.state_dict(), MODEL_PATH)
                model_version += 1
                update_opponent_pool(model_version)
                last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping all workers...", flush=True)
        stop_event.set()
        time.sleep(2)
        
        solver_manager.stop()
        print("C++ workers stopped.")
        
        for worker in inference_workers:
            worker.join(timeout=5)
            if worker.is_alive():
                print(f"Terminating worker {worker.name}...", flush=True)
                worker.terminate()
        print("All Python workers stopped.")
        
        print("Final model saving...", flush=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "paqn_model_final.pth"))
        print("Training finished.")

if __name__ == "__main__":
    main()
