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
import random
import glob

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

sys.path.insert(0, '/content/PAQN/build')
sys.path.insert(0, '/content/PAQN')

from python_src.model import OFC_CNN_Network
from python_src.featurizer import featurize_state_optimal, RANKS, SUITS
from ofc_engine import ReplayBuffer, initialize_evaluator, SolverManager

# --- Настройки ---
TOTAL_CPUS = os.cpu_count() or 1
RESERVED_CPUS = 2 if TOTAL_CPUS > 4 else 1
NUM_INFERENCE_WORKERS = max(1, int(TOTAL_CPUS * 0.20))
NUM_CPP_WORKERS = max(1, TOTAL_CPUS - NUM_INFERENCE_WORKERS - RESERVED_CPUS)

print(f"Configuration: {NUM_CPP_WORKERS} C++ workers, {NUM_INFERENCE_WORKERS} Python inference workers.")

# --- Гиперпараметры ---
ACTION_VECTOR_SIZE = 208
LEARNING_RATE = 0.0005
BUFFER_CAPACITY = 2_000_000
BATCH_SIZE = 2048
MIN_BUFFER_FILL_RATIO = 0.05
MIN_BUFFER_FILL_SAMPLES = int(BUFFER_CAPACITY * MIN_BUFFER_FILL_RATIO)

# --- Пути и интервалы ---
STATS_INTERVAL_SECONDS = 15
SAVE_INTERVAL_SECONDS = 600
MODEL_DIR = "/content/models"
MODEL_PATH = os.path.join(MODEL_DIR, "paqn_model_latest.pth")
OPPONENT_POOL_DIR = os.path.join(MODEL_DIR, "opponent_pool")
MAX_OPPONENTS_IN_POOL = 20

def decode_raw_state(raw_state: list[int]) -> dict:
    s = {}
    ptr = 0
    
    s['street'] = raw_state[ptr]; ptr += 1
    s['current_player'] = raw_state[ptr]; ptr += 1
    player_view = raw_state[ptr]; ptr += 1
    
    hand_size = raw_state[ptr]; ptr += 1
    s['hand'] = [f"{RANKS[c//4]}{SUITS[c%4]}" for c in raw_state[ptr:ptr+hand_size] if c != 255]
    ptr += hand_size

    def read_board(p):
        board = {'top': [], 'middle': [], 'bottom': []}
        for c in raw_state[p:p+3]:
            if c != 255: board['top'].append(f"{RANKS[c//4]}{SUITS[c%4]}")
        p += 3
        for c in raw_state[p:p+5]:
            if c != 255: board['middle'].append(f"{RANKS[c//4]}{SUITS[c%4]}")
        p += 5
        for c in raw_state[p:p+5]:
            if c != 255: board['bottom'].append(f"{RANKS[c//4]}{SUITS[c%4]}")
        p += 5
        return board, p

    my_board, ptr = read_board(ptr)
    opp_board, ptr = read_board(ptr)
    
    s['player_board'] = my_board
    s['opponent_board'] = opp_board

    s['is_player_fantasyland'] = bool(raw_state[ptr]); ptr += 1
    s['is_opponent_fantasyland'] = bool(raw_state[ptr]); ptr += 1
    
    s['player_dead_hands'] = {'top': False, 'middle': False, 'bottom': False}
    s['opponent_dead_hands'] = {'top': False, 'middle': False, 'bottom': False}
    s['is_player_turn'] = s['current_player'] == player_view

    return s

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
        last_model_check_time = time.time()

        while not self.stop_event.is_set():
            try:
                if time.time() - last_model_check_time > 60:
                    self._load_models()
                    last_model_check_time = time.time()

                req_id, is_policy, raw_state, action_vectors = self.task_queue.get(timeout=1.0)
                
                game_dict = decode_raw_state(raw_state)
                state_image = featurize_state_optimal(game_dict)
                
                current_player_id = game_dict['current_player']
                active_model = self.latest_model if current_player_id == 0 else self.opponent_model

                if is_policy:
                    if not action_vectors:
                        predictions = []
                    else:
                        batch_size = len(action_vectors)
                        state_image_tensor = torch.tensor(np.array([state_image] * batch_size), dtype=torch.float32, device=self.device)
                        actions_tensor = torch.tensor(action_vectors, dtype=torch.float32, device=self.device)
                        with torch.inference_mode():
                            policy_logits, _ = active_model(state_image_tensor, actions_tensor)
                            predictions = policy_logits.cpu().numpy().flatten().tolist()
                    self.result_queue.put((req_id, True, predictions))
                else:
                    state_image_tensor = torch.tensor(np.array([state_image]), dtype=torch.float32, device=self.device)
                    with torch.inference_mode():
                        value = active_model(state_image_tensor)
                        prediction = value.item()
                    self.result_queue.put((req_id, False, [prediction]))

            except mp.queues.Empty:
                continue
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
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
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
        NUM_CPP_WORKERS, policy_buffer, value_buffer,
        request_queue, result_queue
    )
    print("C++ workers are running in the background.", flush=True)
    
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)
    last_stats_time = time.time()
    last_save_time = time.time()
    model_version = 0
    
    try:
        while True:
            if value_buffer.size() < MIN_BUFFER_FILL_SAMPLES or policy_buffer.size() < MIN_BUFFER_FILL_SAMPLES:
                time.sleep(1)
                last_stats_time = time.time()
                continue

            model.train()
            optimizer.zero_grad()
            
            # Value Head
            v_batch = value_buffer.sample(BATCH_SIZE)
            v_raw_states, _, v_targets_np = v_batch
            v_state_images = [featurize_state_optimal(decode_raw_state(rs)) for rs in v_raw_states]
            v_infosets = torch.tensor(np.array(v_state_images), device=device)
            v_targets = torch.from_numpy(v_targets_np).unsqueeze(1).to(device)
            pred_values = model(v_infosets)
            loss_v = criterion(pred_values, v_targets)
            
            # Policy Head
            p_batch = policy_buffer.sample(BATCH_SIZE)
            p_raw_states, p_actions_np, p_advantages_np = p_batch
            p_state_images = [featurize_state_optimal(decode_raw_state(rs)) for rs in p_raw_states]
            p_infosets = torch.tensor(np.array(p_state_images), device=device)
            p_actions = torch.from_numpy(p_actions_np).to(device)
            p_advantages = torch.from_numpy(p_advantages_np).unsqueeze(1).to(device)
            pred_logits, _ = model(p_infosets, p_actions)
            loss_p = criterion(pred_logits, p_advantages)

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
                print("\n--- Saving models and updating opponent pool ---", flush=True)
                torch.save(model.state_dict(), MODEL_PATH)
                model_version += 1
                update_opponent_pool(model_version)
                last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping all workers...", flush=True)
        solver_manager.stop()
        print("C++ workers stopped.")
        
        stop_event.set()
        time.sleep(2)
        
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
