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
import subprocess
import shutil

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
build_dir = os.path.join(project_root, 'build')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)


from python_src.model import OFC_CNN_Network
from ofc_engine import ReplayBuffer, initialize_evaluator, SolverManager

# --- КОНСТАНТЫ ---
NUM_FEATURE_CHANNELS = 16
NUM_SUITS = 4
NUM_RANKS = 13
INFOSET_SIZE = NUM_FEATURE_CHANNELS * NUM_SUITS * NUM_RANKS

# --- НАСТРОЙКИ ---
TOTAL_CPUS = os.cpu_count() or 88
RESERVED_CPUS = 4
NUM_INFERENCE_WORKERS = 8
NUM_CPP_WORKERS = 16

print(f"Configuration: {NUM_CPP_WORKERS} C++ workers, {NUM_INFERENCE_WORKERS} Python inference workers.")

# --- ГИПЕРПАРАМЕТРЫ ---
ACTION_LIMIT = 100
LEARNING_RATE = 0.0005
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 256
MIN_BUFFER_FILL_SAMPLES = 50000

# --- ПУТИ И ИНТЕРВАЛЫ ---
STATS_INTERVAL_SECONDS = 15
# ИЗМЕНЕНИЕ: Разделяем интервалы
LOCAL_SAVE_INTERVAL_SECONDS = 300  # 5 минут для локального сохранения
GIT_PUSH_INTERVAL_SECONDS = 600    # 10 минут для пуша на GitHub

# ИЗМЕНЕНИЕ: Локальные модели хранятся в отдельной папке, не отслеживаемой Git
LOCAL_MODEL_DIR = "/content/local_models"
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "paqn_model_latest.pth")
VERSION_FILE = os.path.join(LOCAL_MODEL_DIR, "latest_version.txt")
OPPONENT_POOL_DIR = os.path.join(LOCAL_MODEL_DIR, "opponent_pool")
MAX_OPPONENTS_IN_POOL = 20

# Пути к файлам в репозитории для синхронизации с Git
GIT_MODEL_PATH = os.path.join(project_root, "paqn_model_latest.pth")

# --- НАСТРОЙКИ GIT ---
GIT_REPO_OWNER = "Azerus96"
GIT_REPO_NAME = "PAQN"
GIT_BRANCH = "main"

# --- GIT HELPER FUNCTIONS ---
def run_git_command(command, repo_path):
    try:
        result = subprocess.run(command, cwd=repo_path, check=True, capture_output=True, text=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {' '.join(command)}\nError: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Git command timed out: {' '.join(command)}")
        return False

def git_push(commit_message, repo_path, auth_repo_url):
    print(f"\n--- Attempting to push to GitHub: '{commit_message}' ---")
    
    # Копируем последнюю локальную модель в репозиторий для пуша
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, GIT_MODEL_PATH)
        print(f"Copied {MODEL_PATH} to {GIT_MODEL_PATH} for pushing.")
    else:
        print("No local model to push.")
        return

    if not run_git_command(["git", "add", GIT_MODEL_PATH], repo_path): return
    
    status_result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_path, capture_output=True, text=True)
    if not status_result.stdout.strip():
        print("No changes to commit.")
        return

    if not run_git_command(["git", "commit", "-m", commit_message], repo_path): return
    if not run_git_command(["git", "push", auth_repo_url, f"HEAD:{GIT_BRANCH}"], repo_path): return
    print("--- Push successful ---")

def git_pull(repo_path, auth_repo_url):
    print("\n--- Pulling latest model from GitHub ---")
    if not run_git_command(["git", "pull", auth_repo_url, GIT_BRANCH], repo_path):
        print("Git pull failed. Continuing with local version if available.")

# ... Класс InferenceWorker остается БЕЗ ИЗМЕНЕНИЙ ...
class InferenceWorker(mp.Process):
    def __init__(self, name, task_queue, result_dict, log_queue, stop_event):
        super().__init__(name=name)
        self.task_queue = task_queue
        self.result_dict = result_dict
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.latest_model = None
        self.opponent_model = None
        self.device = None
        self.opponent_pool_files = []
        self.model_version = -1
        self.last_version_check_time = 0
        self.request_counter = 0

    def _log(self, message):
        self.log_queue.put(f"[{self.name}] {message}")

    def _initialize(self):
        self._log("Started.")
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
                self._log(f"Loaded latest model (version {self.model_version}).")
            else:
                self._log("No latest model found, using initialized weights.")

            self.opponent_pool_files = glob.glob(os.path.join(OPPONENT_POOL_DIR, "*.pth"))
            if self.opponent_pool_files:
                opponent_path = random.choice(self.opponent_pool_files)
                self.opponent_model.load_state_dict(torch.load(opponent_path, map_location=self.device))
                self._log(f"Loaded opponent model: {os.path.basename(opponent_path)}")
            else:
                self.opponent_model.load_state_dict(self.latest_model.state_dict())
                self._log("Opponent pool is empty, using latest model as opponent.")

        except Exception as e:
            self._log(f"!!! FAILED to load models: {e}")
            traceback.print_exc()

    def _check_for_updates(self):
        self.request_counter += 1
        if self.request_counter % 100 != 0 and time.time() - self.last_version_check_time < 5:
            return

        self.last_version_check_time = time.time()
        try:
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, 'r') as f:
                    latest_version = int(f.read())
                if latest_version > self.model_version:
                    self._log(f"New model version detected ({latest_version}). Reloading models...")
                    self.model_version = latest_version
                    self._load_models()
        except (IOError, ValueError) as e:
            self._log(f"Could not check for model update: {e}")


    def run(self):
        self._initialize()
        
        STREET_START_IDX = 9
        STREET_END_IDX = 14
        TURN_CHANNEL_IDX = 15

        while not self.stop_event.is_set():
            try:
                try:
                    req_id, is_policy, infoset, action_vectors = self.task_queue.get(timeout=1)
                except queue.Empty:
                    self._check_for_updates()
                    continue

                self._check_for_updates()
                
                with torch.inference_mode():
                    infoset_tensor = torch.tensor([infoset], dtype=torch.float32, device=self.device)
                    infoset_tensor = infoset_tensor.view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS)
                    
                    turn_channel_val = infoset[TURN_CHANNEL_IDX * NUM_SUITS * NUM_RANKS]
                    is_player_turn = turn_channel_val > 0.5
                    model_to_use = self.latest_model if is_player_turn else self.opponent_model

                    if is_policy:
                        if not action_vectors:
                            result = (req_id, True, [])
                        else:
                            num_actions = len(action_vectors)
                            infoset_batch = infoset_tensor.repeat(num_actions, 1, 1, 1)
                            actions_tensor = torch.tensor(action_vectors, dtype=torch.float32, device=self.device)
                            street_vector = infoset_batch[:, STREET_START_IDX:STREET_END_IDX, 0, 0]
                            
                            policy_logits, _ = model_to_use(infoset_batch, actions_tensor, street_vector)
                            predictions = policy_logits.cpu().numpy().flatten().tolist()
                            result = (req_id, True, predictions)
                    else: # is_value
                        value = model_to_use(infoset_tensor)
                        result = (req_id, False, [value.item()])
                    
                    self.result_dict[req_id] = result

            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                self._log(f"---!!! EXCEPTION IN {self.name} !!!---")
                exc_info = traceback.format_exc()
                for line in exc_info.split('\n'):
                    self._log(line)
        
        self._log("Stopped.")

def update_opponent_pool(model_version):
    if not os.path.exists(MODEL_PATH): return
    state_dict_to_save = torch.load(MODEL_PATH)
    
    os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
    pool_files = sorted(glob.glob(os.path.join(OPPONENT_POOL_DIR, "*.pth")), key=os.path.getmtime)
    
    while len(pool_files) >= MAX_OPPONENTS_IN_POOL:
        os.remove(pool_files.pop(0))
        
    new_opponent_path = os.path.join(OPPONENT_POOL_DIR, f"paqn_model_v{model_version}.pth")
    torch.save(state_dict_to_save, new_opponent_path)
    print(f"Added model version {model_version} to opponent pool.")


def main():
    # --- АУТЕНТИФИКАЦИЯ И НАСТРОЙКА GIT ---
    git_username = os.environ.get('GIT_USERNAME')
    git_token = os.environ.get('GIT_TOKEN')

    if not git_username or not git_token:
        print("ERROR: GIT_USERNAME and GIT_TOKEN environment variables must be set.")
        sys.exit(1)
        
    auth_repo_url = f"https://{git_username}:{git_token}@github.com/{GIT_REPO_OWNER}/{GIT_REPO_NAME}.git"
    
    run_git_command(["git", "config", "--global", "user.email", f"{git_username}@users.noreply.github.com"], project_root)
    run_git_command(["git", "config", "--global", "user.name", git_username], project_root)
    
    git_pull(project_root, auth_repo_url)
    
    # Создаем локальные директории
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)

    # Копируем модель из Git в локальную директорию, если она есть
    if os.path.exists(GIT_MODEL_PATH) and not os.path.exists(MODEL_PATH):
        print(f"Copying model from Git repo to local directory: {GIT_MODEL_PATH} -> {MODEL_PATH}")
        shutil.copy2(GIT_MODEL_PATH, MODEL_PATH)
    
    print("Initializing C++ hand evaluator lookup tables...", flush=True)
    initialize_evaluator()
    print("C++ evaluator initialized successfully.", flush=True)

    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    
    model = OFC_CNN_Network().to(device)
    model_version = 0
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Loaded latest model from local path:", MODEL_PATH)
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, 'r') as f:
                    model_version = int(f.read())
                print(f"Current model version: {model_version}")
        except Exception as e:
            print(f"Could not load model, starting from scratch. Error: {e}")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    policy_buffer = ReplayBuffer(BUFFER_CAPACITY)
    value_buffer = ReplayBuffer(BUFFER_CAPACITY)
    
    manager = mp.Manager()
    request_queue = manager.Queue(maxsize=NUM_CPP_WORKERS * 16)
    result_dict = manager.dict()
    log_queue = manager.Queue()
    stop_event = mp.Event()

    inference_workers = []
    for i in range(NUM_INFERENCE_WORKERS):
        worker = InferenceWorker(f"InferenceWorker-{i}", request_queue, result_dict, log_queue, stop_event)
        worker.start()
        inference_workers.append(worker)

    print(f"Creating C++ SolverManager with {NUM_CPP_WORKERS} workers...", flush=True)
    solver_manager = SolverManager(
        NUM_CPP_WORKERS, ACTION_LIMIT, policy_buffer, value_buffer,
        request_queue, result_dict, log_queue
    )
    
    solver_manager.start()
    print("C++ workers are running in the background.", flush=True)
    
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)
    last_stats_time = time.time()
    last_local_save_time = time.time()
    last_git_push_time = time.time()
    
    STREET_START_IDX = 9
    STREET_END_IDX = 14

    try:
        while True:
            if time.time() - last_stats_time > STATS_INTERVAL_SECONDS:
                while not log_queue.empty():
                    try:
                        print(log_queue.get(timeout=0.01), flush=True)
                    except queue.Empty:
                        break
                
                total_generated = policy_buffer.total_generated()
                avg_p_loss = np.mean(policy_losses) if policy_losses else float('nan')
                avg_v_loss = np.mean(value_losses) if value_losses else float('nan')
                
                print("\n" + "="*20 + " STATS UPDATE " + "="*20, flush=True)
                print(f"Time: {time.strftime('%H:%M:%S')}", flush=True)
                print(f"Model Version: {model_version}", flush=True)
                print(f"Total Generated: {total_generated:,}", flush=True)
                print(f"Buffer Fill -> Policy: {policy_buffer.size():,}/{BUFFER_CAPACITY:,} ({policy_buffer.size()/BUFFER_CAPACITY:.1%}) "
                      f"| Value: {value_buffer.size():,}/{BUFFER_CAPACITY:,} ({value_buffer.size()/BUFFER_CAPACITY:.1%})", flush=True)
                print(f"Avg Losses (last 100) -> Policy: {avg_p_loss:.6f} | Value: {avg_v_loss:.6f}", flush=True)
                print(f"Request Queue: {request_queue.qsize()} | Result Dict: {len(result_dict)}", flush=True)
                print("="*54, flush=True)
                last_stats_time = time.time()

            if value_buffer.size() < MIN_BUFFER_FILL_SAMPLES or policy_buffer.size() < MIN_BUFFER_FILL_SAMPLES:
                print(f"Waiting for buffer to fill... Policy: {policy_buffer.size()}/{MIN_BUFFER_FILL_SAMPLES} | Value: {value_buffer.size()}/{MIN_BUFFER_FILL_SAMPLES}", end='\r', flush=True)
                time.sleep(1)
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
            if now - last_local_save_time > LOCAL_SAVE_INTERVAL_SECONDS:
                print("\n--- Saving models locally and updating opponent pool ---", flush=True)
                torch.save(model.state_dict(), MODEL_PATH)
                model_version += 1
                with open(VERSION_FILE, 'w') as f:
                    f.write(str(model_version))
                update_opponent_pool(model_version)
                last_local_save_time = now
            
            if now - last_git_push_time > GIT_PUSH_INTERVAL_SECONDS:
                git_push(f"Periodic model save v{model_version}", project_root, auth_repo_url)
                last_git_push_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping all workers...", flush=True)
        stop_event.set()
        
        print("Stopping C++ workers...", flush=True)
        solver_manager.stop()
        print("C++ workers stopped.")
        
        print("Stopping Python workers...", flush=True)
        for worker in inference_workers:
            worker.join(timeout=5)
            if worker.is_alive():
                print(f"Terminating worker {worker.name}...", flush=True)
                worker.terminate()
        print("All Python workers stopped.")
        
        print("Final model saving...", flush=True)
        torch.save(model.state_dict(), MODEL_PATH)
        git_push(f"Final model save v{model_version} on exit", project_root, auth_repo_url)
        
        print("Training finished.")

if __name__ == "__main__":
    main()
