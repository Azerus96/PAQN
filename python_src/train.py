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
from collections import deque

# --- НАСТРОЙКИ ---
# Используйте меньше воркеров для отладки, если нужно
NUM_WORKERS = int(os.cpu_count() or 88) 
NUM_COMPUTATION_THREADS = "8"
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

from .model import ParametricDuelingNetwork
from ofc_engine import DeepMCCFR, SharedReplayBuffer, InferenceQueue

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_VECTOR_SIZE = 208
ACTION_LIMIT = 1000
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 4096
SAVE_INTERVAL_SECONDS = 300
MODEL_PATH = "paqn_d2cfr_model.pth"

# Параметры для пакетного инференса
INFERENCE_BATCH_SIZE = 2048 # Максимальное количество действий в одном батче для нейросети
INFERENCE_MAX_DELAY_MS = 5 # Максимальная задержка в мс для сбора батча

class InferenceWorker(threading.Thread):
    def __init__(self, model, queue, device):
        super().__init__(daemon=True)
        self.model = model
        self.queue = queue
        self.device = device
        self.stop_event = threading.Event()

    def run(self):
        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) started.", flush=True)
        self.model.eval() # Переводим модель в режим инференса

        while not self.stop_event.is_set():
            try:
                # Ждем первого запроса
                self.queue.wait() 
                
                # Собираем батч запросов
                requests = self.queue.pop_all()
                
                if not requests:
                    continue

                # --- Логика пакетной обработки ---
                infoset_batch = []
                action_batch = []
                # Структура для отслеживания, какому запросу какая часть батча принадлежит
                request_indices = [] 

                total_actions = 0
                for req in requests:
                    num_actions = len(req.action_vectors)
                    if num_actions == 0:
                        # Пустой запрос, который нужно обработать
                        request_indices.append(0)
                        continue

                    # Расширяем инфосет для каждого действия
                    infoset_batch.extend([req.infoset] * num_actions)
                    action_batch.extend(req.action_vectors)
                    request_indices.append(num_actions)
                    total_actions += num_actions

                if total_actions == 0:
                    # Обрабатываем пустые запросы
                    for i, req in enumerate(requests):
                         if request_indices[i] == 0:
                            req.set_result([])
                    continue

                # Преобразуем в тензоры и делаем инференс
                infosets_tensor = torch.tensor(infoset_batch, dtype=torch.float32, device=self.device)
                actions_tensor = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    predictions = self.model(infosets_tensor, actions_tensor).cpu().numpy().flatten()

                # Разбираем результаты и отправляем обратно
                start_idx = 0
                for i, req in enumerate(requests):
                    num_actions = request_indices[i]
                    if num_actions == 0:
                        req.set_result([])
                        continue
                    
                    end_idx = start_idx + num_actions
                    result = predictions[start_idx:end_idx].tolist()
                    req.set_result(result)
                    start_idx = end_idx

            except Exception as e:
                print(f"Error in InferenceWorker: {e}", flush=True)
                traceback.print_exc()
        
        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) stopped.", flush=True)

    def stop(self):
        self.stop_event.set()

# --- ИСПРАВЛЕННАЯ ФУНКЦИЯ ---
def push_to_github(model_path, commit_message):
    """
    Безопасно отправляет изменения на GitHub.
    Добавляет все отслеживаемые файлы и создает коммит, только если есть изменения.
    """
    try:
        print("Pushing progress to GitHub...", flush=True)
        # Настройка пользователя Git для коммита
        subprocess.run(['git', 'config', '--global', 'user.email', 'bot@example.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Training Bot'], check=True)
        
        # Добавляем ВСЕ измененные и новые файлы в текущей директории
        subprocess.run(['git', 'add', '.'], check=True)
        
        # Проверяем, есть ли что коммитить, чтобы избежать пустых коммитов
        status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        
        if status_result.stdout:
            print("Changes detected, creating commit...")
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            subprocess.run(['git', 'push'], check=True)
            print("Progress pushed successfully.", flush=True)
        else:
            print("No changes to commit. Skipping push.", flush=True)
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub: {e}", flush=True)
        print(f"Stderr: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred during git push: {e}", flush=True)

def main():
    # Используем GPU, если доступен, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = ParametricDuelingNetwork(
        infoset_size=INPUT_SIZE, 
        action_vec_size=ACTION_VECTOR_SIZE
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}. Loading weights...", flush=True)
        try:
            # Загружаем веса на нужное устройство
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            print(f"Could not load state_dict. Error: {e}. Starting from scratch.", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY)
    inference_queue = InferenceQueue()

    # Передаем модель и устройство в воркер
    inference_worker = InferenceWorker(model, inference_queue, device)
    inference_worker.start()

    solvers = [DeepMCCFR(ACTION_LIMIT, replay_buffer, inference_queue) for _ in range(NUM_WORKERS)]
    
    stop_event = threading.Event()
    git_thread = None
    training_losses = deque(maxlen=100) # Для сглаживания лосса

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            def worker_loop(solver):
                while not stop_event.is_set():
                    solver.run_traversal()

            print(f"Submitting {NUM_WORKERS} long-running C++ worker tasks...", flush=True)
            futures = {executor.submit(worker_loop, s) for s in solvers}
            
            last_save_time = time.time()
            last_report_time = time.time()
            last_report_head = 0
            
            while True:
                time.sleep(0.1)
                
                current_buffer_size = replay_buffer.get_count()
                
                if current_buffer_size >= BATCH_SIZE:
                    model.train() # Переводим модель в режим обучения
                    
                    infosets_np, actions_np, targets_np = replay_buffer.sample(BATCH_SIZE)
                    
                    infosets = torch.from_numpy(infosets_np).to(device)
                    actions = torch.from_numpy(actions_np).to(device)
                    targets = torch.from_numpy(targets_np).to(device)

                    optimizer.zero_grad()
                    
                    predictions = model(infosets, actions)
                    
                    loss = criterion(predictions, targets)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    training_losses.append(loss.item())
                    
                    # Возвращаем модель в режим инференса для потока-инференсера
                    model.eval()
                
                now = time.time()
                if now - last_report_time > 10.0:
                    duration = now - last_report_time
                    current_head = replay_buffer.get_head()
                    samples_generated_interval = current_head - last_report_head
                    last_report_head = current_head
                    
                    samples_per_sec = samples_generated_interval / duration if duration > 0 else 0
                    
                    print(f"\n--- Stats Update ---", flush=True)
                    print(f"Throughput: {samples_per_sec:.2f} samples/s. Buffer: {current_buffer_size}/{REPLAY_BUFFER_CAPACITY}. Total generated: {current_head:,}", flush=True)
                    
                    if training_losses:
                        avg_loss = sum(training_losses) / len(training_losses)
                        print(f"Avg training loss (last 100): {avg_loss:.6f}", flush=True)

                    last_report_time = now

                    if now - last_save_time > SAVE_INTERVAL_SECONDS:
                        if git_thread and git_thread.is_alive():
                            print("Previous Git push is still running. Skipping this save.", flush=True)
                        else:
                            if training_losses:
                                print("\n--- Saving model and pushing to GitHub ---", flush=True)
                                torch.save(model.state_dict(), MODEL_PATH)
                                avg_loss = sum(training_losses) / len(training_losses)
                                commit_message = f"PAQN Training. Samples: {current_head:,}. Loss: {avg_loss:.6f}"
                                
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
