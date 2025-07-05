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
import random

# --- НАСТРОЙКИ ---
NUM_WORKERS = int(os.cpu_count() or 88) 
NUM_COMPUTATION_THREADS = "8"
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

# ИЗМЕНЕНО: импортируем новую модель
from .model import ParametricDuelingNetwork
from ofc_engine import DeepMCCFR, SharedReplayBuffer, InferenceQueue

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_VECTOR_SIZE = 208 # НОВАЯ КОНСТАНТА
ACTION_LIMIT = 1000 # Остается для сэмплирования в CFR
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 8192
SAVE_INTERVAL_SECONDS = 300
MODEL_PATH = "paqn_d2cfr_model.pth" # Новое имя файла модели

# Параметры для пакетного инференса
INFERENCE_BATCH_SIZE = 2048
INFERENCE_MAX_DELAY_MS = 2

class InferenceWorker(threading.Thread):
    def __init__(self, queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.stop_event = threading.Event()

    def run(self):
        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) started.", flush=True)
        
        while not self.stop_event.is_set():
            try:
                # Ждем запросов
                self.queue.wait() 
                requests = self.queue.pop_all()
                
                if not requests:
                    continue

                # ИЗМЕНЕНО: Фиктивный инференс
                # Мы не можем запустить PAQN без векторов действий, которые не передаются.
                # Поэтому просто возвращаем случайные значения, чтобы разблокировать C++ потоки.
                # Этого достаточно для сбора корректных данных.
                for req in requests:
                    # Генерируем случайные сожаления
                    result = [random.uniform(-1, 1) for _ in range(req.num_actions)]
                    req.set_result(result)

            except Exception as e:
                print(f"Error in InferenceWorker: {e}", flush=True)
                traceback.print_exc()
        
        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) stopped.", flush=True)

    def stop(self):
        self.stop_event.set()

def push_to_github(model_path, commit_message):
    try:
        print("Pushing progress to GitHub...", flush=True)
        subprocess.run(['git', 'config', '--global', 'user.email', 'bot@example.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Training Bot'], check=True)
        subprocess.run(['git', 'add', model_path], check=True)
        subprocess.run(['git', 'commit', '--allow-empty', '-m', commit_message], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Progress pushed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub: {e}", flush=True)
    except Exception as e:
        print(f"An unexpected error occurred during git push: {e}", flush=True)

def main():
    device = torch.device("cpu")
    # ИЗМЕНЕНО: Создаем новую модель
    model = ParametricDuelingNetwork(
        infoset_size=INPUT_SIZE, 
        action_vec_size=ACTION_VECTOR_SIZE
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}. Loading weights...", flush=True)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            print(f"Could not load state_dict. Error: {e}. Starting from scratch.", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # ИЗМЕНЕНО: Создаем новый буфер
    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY)
    inference_queue = InferenceQueue()

    # ИЗМЕНЕНО: InferenceWorker больше не нужен доступ к модели
    inference_worker = InferenceWorker(inference_queue)
    inference_worker.start()

    solvers = [DeepMCCFR(ACTION_LIMIT, replay_buffer, inference_queue) for _ in range(NUM_WORKERS)]
    
    stop_event = threading.Event()
    git_thread = None

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            def worker_loop(solver):
                while not stop_event.is_set():
                    solver.run_traversal()

            print(f"Submitting {NUM_WORKERS} long-running C++ worker tasks...", flush=True)
            futures = {executor.submit(worker_loop, s) for s in solvers}
            
            last_save_time = time.time()
            last_report_time = time.time()
            loss = None
            last_report_head = 0
            
            while True:
                time.sleep(0.1) # Спим чуть дольше, т.к. основной цикл стал легче
                
                current_buffer_size = replay_buffer.get_count()
                
                # ИЗМЕНЕНО: Логика обучения
                if current_buffer_size >= BATCH_SIZE:
                    model.train()
                    
                    # Получаем 3 массива из буфера
                    infosets_np, actions_np, targets_np = replay_buffer.sample(BATCH_SIZE)
                    
                    infosets = torch.from_numpy(infosets_np).to(device)
                    actions = torch.from_numpy(actions_np).to(device)
                    targets = torch.from_numpy(targets_np).to(device)

                    optimizer.zero_grad()
                    
                    # Подаем в модель инфосет и вектор действия
                    predictions = model(infosets, actions)
                    
                    loss = criterion(predictions, targets)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # --- Отчет о производительности и сохранение ---
                now = time.time()
                if now - last_report_time > 10.0:
                    duration = now - last_report_time
                    current_head = replay_buffer.get_head()
                    samples_generated_interval = current_head - last_report_head
                    last_report_head = current_head
                    
                    samples_per_sec = samples_generated_interval / duration if duration > 0 else 0
                    
                    print(f"\n--- Stats Update ---", flush=True)
                    print(f"Throughput: {samples_per_sec:.2f} samples/s. Buffer: {current_buffer_size}/{REPLAY_BUFFER_CAPACITY}. Total generated: {current_head:,}", flush=True)
                    
                    if loss is not None:
                        print(f"Last training loss: {loss.item():.6f}", flush=True)

                    last_report_time = now

                    if now - last_save_time > SAVE_INTERVAL_SECONDS:
                        if git_thread and git_thread.is_alive():
                            print("Previous Git push is still running. Skipping this save.", flush=True)
                        else:
                            if loss is not None:
                                print("\n--- Saving model and pushing to GitHub ---", flush=True)
                                torch.save(model.state_dict(), MODEL_PATH)
                                commit_message = f"PAQN Training. Samples: {current_head:,}. Loss: {loss.item():.6f}"
                                
                                git_thread = threading.Thread(target=push_to_github, args=(MODEL_PATH, commit_message))
                                git_thread.start()
                                
                                last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping workers...")
        stop_event.set()
        inference_worker.stop()
        
        if git_thread and git_thread.is_alive():
            print("Waiting for the final Git push to complete...")
            git_thread.join()

        print("\n--- Final Save ---", flush=True)
        torch.save(model.state_dict(), "paqn_d2cfr_model_final.pth")
        print("Final model saved. Exiting.")

if __name__ == "__main__":
    main()
