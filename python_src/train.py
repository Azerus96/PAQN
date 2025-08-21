# --- НАЧАЛО ФАЙЛА python_src/train.py ---

# --- ШАГ 0: УСТАНОВКА ЛИМИТОВ ПОТОКОВ ДО ВСЕХ ИМПОРТОВ ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")

import sys
import time
import torch
# --- УСТАНОВКА ЛИМИТОВ ПОТОКОВ ДЛЯ PYTORCH ---
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn.functional as F
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
import gc
import psutil
import threading
import aim

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
NUM_INFERENCE_WORKERS = 24
NUM_CPP_WORKERS = 48

print(f"Configuration: {NUM_CPP_WORKERS} C++ workers, {NUM_INFERENCE_WORKERS} Python inference workers.")

# --- ГИПЕРПАРАМЕТРЫ ---
ACTION_LIMIT = 100
LEARNING_RATE = 0.0005
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 512
MIN_BUFFER_FILL_SAMPLES = 50000

# --- ПУТИ И ИНТЕРВАЛЫ ---
STATS_INTERVAL_SECONDS = 15
LOCAL_SAVE_INTERVAL_SECONDS = 300
GIT_PUSH_INTERVAL_SECONDS = 600

LOCAL_MODEL_DIR = "/content/local_models"
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "paqn_model_latest.pth")
VERSION_FILE = os.path.join(LOCAL_MODEL_DIR, "latest_version.txt")
OPPONENT_POOL_DIR = os.path.join(LOCAL_MODEL_DIR, "opponent_pool")
MAX_OPPONENTS_IN_POOL = 20

GIT_MODEL_PATH = os.path.join(project_root, "paqn_model_latest.pth")

# --- НАСТРОЙКИ GIT ---
GIT_REPO_OWNER = "Azerus96"
GIT_REPO_NAME = "PAQN"
GIT_BRANCH = "main"

# --- GIT HELPER FUNCTIONS (без изменений) ---
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
                for attempt in range(3):
                    try:
                        self.latest_model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                        self._log(f"Loaded latest model (version {self.model_version}).")
                        break
                    except Exception as e:
                        self._log(f"Attempt {attempt+1} to load latest model failed: {e}. Retrying in 0.1s...")
                        time.sleep(0.1)
                else:
                    self._log(f"!!! CRITICAL: FAILED to load latest model after 3 attempts. Continuing with old version.")
            else:
                self._log("No latest model found, using initialized weights.")
        except Exception as e:
            self._log(f"!!! UNEXPECTED EXCEPTION during latest model loading: {e}")
            traceback.print_exc()

        try:
            self.opponent_pool_files = glob.glob(os.path.join(OPPONENT_POOL_DIR, "*.pth"))
            if self.opponent_pool_files:
                opponent_path = random.choice(self.opponent_pool_files)
                for attempt in range(3):
                    try:
                        self.opponent_model.load_state_dict(torch.load(opponent_path, map_location=self.device))
                        self._log(f"Loaded opponent model: {os.path.basename(opponent_path)}")
                        break
                    except Exception as e:
                        self._log(f"Attempt {attempt+1} to load opponent model failed: {e}. Retrying in 0.1s...")
                        time.sleep(0.1)
                else:
                    self._log(f"!!! CRITICAL: FAILED to load opponent model. Using latest model as opponent.")
                    self.opponent_model.load_state_dict(self.latest_model.state_dict())
            else:
                self.opponent_model.load_state_dict(self.latest_model.state_dict())
                self._log("Opponent pool is empty, using latest model as opponent.")
        except Exception as e:
            self._log(f"!!! UNEXPECTED EXCEPTION during opponent model loading: {e}")
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
                    worker_id = int(self.name.split('-')[-1])
                    time.sleep(worker_id * 0.1)
                    
                    self._log(f"New model version detected ({latest_version}). Reloading models...")
                    self.model_version = latest_version
                    self._load_models()
        except (IOError, ValueError) as e:
            self._log(f"Could not check for model update: {e}")

    def run(self):
        self._initialize()
        
        STREET_START_IDX = 9
        STREET_END_IDX = 14

        while not self.stop_event.is_set():
            try:
                try:
                    request_tuple = self.task_queue.get(timeout=1)
                    req_id, is_policy, infoset, action_vectors, is_traverser_turn = request_tuple
                except queue.Empty:
                    self._check_for_updates()
                    continue

                if self.request_counter % 500 == 1:
                    self._log("--- INPUT TENSOR DIAGNOSTICS ---")
                    infoset_np = np.array(infoset)
                    self._log(f"Infoset stats: shape={infoset_np.shape}, min={infoset_np.min():.2f}, max={infoset_np.max():.2f}, mean={infoset_np.mean():.4f}, non-zero={np.count_nonzero(infoset_np)}")
                    if is_policy and action_vectors:
                        actions_np = np.array(action_vectors)
                        self._log(f"Actions stats: shape={actions_np.shape}, min={actions_np.min():.2f}, max={actions_np.max():.2f}, mean={actions_np.mean():.4f}, non-zero={np.count_nonzero(actions_np)}")
                    self._log("------------------------------------")

                self._check_for_updates()
                
                with torch.inference_mode():
                    infoset_tensor = torch.tensor([infoset], dtype=torch.float32, device=self.device)
                    infoset_tensor = infoset_tensor.view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS)
                    
                    model_to_use = self.latest_model if is_traverser_turn else self.opponent_model

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
    
    os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
    
    temp_opponent_path = os.path.join(OPPONENT_POOL_DIR, f"temp_paqn_model_v{model_version}.pth")
    new_opponent_path = os.path.join(OPPONENT_POOL_DIR, f"paqn_model_v{model_version}.pth")
    
    try:
        shutil.copy2(MODEL_PATH, temp_opponent_path)
        os.rename(temp_opponent_path, new_opponent_path)
        print(f"Added model version {model_version} to opponent pool.")
    except Exception as e:
        print(f"Error updating opponent pool: {e}")
        if os.path.exists(temp_opponent_path):
            os.remove(temp_opponent_path)
        return

    pool_files = sorted(glob.glob(os.path.join(OPPONENT_POOL_DIR, "*.pth")), key=os.path.getmtime)
    
    while len(pool_files) >= MAX_OPPONENTS_IN_POOL:
        try:
            os.remove(pool_files.pop(0))
        except OSError as e:
            print(f"Warning: Could not remove old opponent file: {e}")

def main():
    aim_run = aim.Run(experiment="paqn_ofc_poker")
    aim_run["hparams"] = {
        "num_cpp_workers": NUM_CPP_WORKERS,
        "num_inference_workers": NUM_INFERENCE_WORKERS,
        "learning_rate": LEARNING_RATE,
        "buffer_capacity": BUFFER_CAPACITY,
        "batch_size": BATCH_SIZE,
        "action_limit": ACTION_LIMIT
    }

    def monitor_resources():
        p = psutil.Process(os.getpid())
        while True:
            try:
                rss_gb = p.memory_info().rss / 1024**3
                threads = p.num_threads()
                print(f"[MONITOR] RSS={rss_gb:.2f} GB, Threads={threads}", flush=True)
                aim_run.track(rss_gb, name="system/memory_rss_gb")
                aim_run.track(threads, name="system/num_threads")
                time.sleep(15)
            except (psutil.NoSuchProcess, KeyboardInterrupt):
                break
    threading.Thread(target=monitor_resources, daemon=True).start()

    git_username = os.environ.get('GIT_USERNAME')
    git_token = os.environ.get('GIT_TOKEN')

    if not git_username or not git_token:
        print("ERROR: GIT_USERNAME and GIT_TOKEN environment variables must be set.")
        sys.exit(1)
        
    auth_repo_url = f"https://{git_username}:{git_token}@github.com/{GIT_REPO_OWNER}/{GIT_REPO_NAME}.git"
    
    run_git_command(["git", "config", "--global", "user.email", f"{git_username}@users.noreply.github.com"], project_root)
    run_git_command(["git", "config", "--global", "user.name", git_username], project_root)
    
    git_pull(project_root, auth_repo_url)
    
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)

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

    # !!! ИЗМЕНЕНИЕ: Улучшенная настройка оптимизатора с AdamW и группировкой параметров !!!
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Исключаем смещения (bias) и параметры нормализации (weight/bias) из weight decay
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    print("Optimizer configured with AdamW and parameter groups (with/without weight decay).")
    
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
    last_cleanup_time = time.time()
    
    STREET_START_IDX = 9
    STREET_END_IDX = 14
    global_step = 0

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
                
                aim_run.track(total_generated, name="system/total_samples_generated")
                aim_run.track(policy_buffer.size(), name="buffer/policy_buffer_size")
                aim_run.track(value_buffer.size(), name="buffer/value_buffer_size")
                aim_run.track(request_queue.qsize(), name="system/request_queue_size")
                if policy_losses: aim_run.track(avg_p_loss, name="loss/policy_loss_avg")
                if value_losses: aim_run.track(avg_v_loss, name="loss/value_loss_avg")
                
                last_stats_time = time.time()

            if time.time() - last_cleanup_time > 60:
                if len(result_dict) > 10000:
                    keys = sorted(result_dict.keys())
                    for key in keys[:-5000]:
                        try:
                            del result_dict[key]
                        except KeyError:
                            pass
                    print(f"[CLEANUP] result_dict cleaned. Size before: {len(keys)}, after: {len(result_dict)}", flush=True)
                gc.collect()
                last_cleanup_time = time.time()

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
            
            v_targets_clipped = torch.clamp(v_targets, -50.0, 50.0)
            pred_values = model(v_infosets)
            loss_v = F.huber_loss(pred_values, v_targets_clipped, delta=1.0)
            
            p_batch = policy_buffer.sample(BATCH_SIZE)
            if not p_batch: continue
            p_infosets_np, p_actions_np, p_advantages_np = p_batch
            p_infosets = torch.from_numpy(p_infosets_np).view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS).to(device)
            p_actions = torch.from_numpy(p_actions_np).to(device)
            p_advantages = torch.from_numpy(p_advantages_np).unsqueeze(1).to(device)
            
            adv_mean, adv_std = p_advantages.mean(), p_advantages.std()
            p_advantages_normalized = torch.clamp((p_advantages - adv_mean) / (adv_std + 1e-6), -5.0, 5.0)

            p_street_vector = p_infosets[:, STREET_START_IDX:STREET_END_IDX, 0, 0]
            pred_logits, _ = model(p_infosets, p_actions, p_street_vector)
            loss_p = F.huber_loss(pred_logits, p_advantages_normalized, delta=1.0)

            optimizer.zero_grad()
            total_loss = loss_v + loss_p
            total_loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            value_losses.append(loss_v.item())
            policy_losses.append(loss_p.item())
            global_step += 1
            
            aim_run.track(loss_v.item(), name="loss/value_loss", step=global_step)
            aim_run.track(loss_p.item(), name="loss/policy_loss", step=global_step)
            aim_run.track(grad_norm.item(), name="diagnostics/grad_norm", step=global_step)
            aim_run.track(adv_mean.item(), name="targets/advantage_mean", step=global_step)
            aim_run.track(adv_std.item(), name="targets/advantage_std", step=global_step)
            
            model.eval()

            now = time.time()
            if now - last_local_save_time > LOCAL_SAVE_INTERVAL_SECONDS:
                print("\n--- Saving models locally and updating opponent pool ---", flush=True)
                
                temp_model_path = MODEL_PATH + ".tmp"
                torch.save(model.state_dict(), temp_model_path)
                os.rename(temp_model_path, MODEL_PATH)

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
        temp_model_path = MODEL_PATH + ".tmp"
        torch.save(model.state_dict(), temp_model_path)
        os.rename(temp_model_path, MODEL_PATH)
        git_push(f"Final model save v{model_version} on exit", project_root, auth_repo_url)
        
        aim_run.close()
        print("Training finished.")

if __name__ == "__main__":
    main()

# --- КОНЕЦ ФАЙЛА python_src/train.py ---
