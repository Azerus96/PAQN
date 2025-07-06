#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <functional>
#include "DeepMCCFR.hpp"
#include "SharedReplayBuffer.hpp"
#include "hand_evaluator.hpp"

// Пустые callback-функции, так как у нас нет Python для их обработки.
// C++ код с заглушками будет их вызывать, но они ничего не будут делать.
void empty_policy_cb(uint64_t, const std::vector<float>&, const std::vector<std::vector<float>>&, const std::function<void(std::vector<float>)>& responder) {
    responder({}); // Просто отвечаем пустым вектором
}

void empty_value_cb(uint64_t, const std::vector<float>&, const std::function<void(float)>& responder) {
    responder(0.0f); // Просто отвечаем нулем
}

// Функция, которую будет выполнять каждый C++ поток
void worker_thread_loop(ofc::SharedReplayBuffer* policy_buffer, ofc::SharedReplayBuffer* value_buffer, std::atomic<bool>* stop_flag) {
    // Создаем свой собственный экземпляр DeepMCCFR внутри потока
    ofc::DeepMCCFR solver(1000, policy_buffer, value_buffer, empty_policy_cb, empty_value_cb);
    while (!(*stop_flag)) {
        solver.run_traversal();
    }
}

int main() {
    std::cout << "--- Pure C++ Multi-threading Test ---" << std::endl;

    // 1. Инициализируем все, что нужно
    omp::HandEvaluator::initialize();
    std::cout << "HandEvaluator initialized." << std::endl;

    const int NUM_THREADS = 16; // Начнем с умеренного количества потоков
    const int TEST_DURATION_SECONDS = 30;

    // 2. Создаем общие ресурсы
    ofc::SharedReplayBuffer policy_buffer(2000000);
    ofc::SharedReplayBuffer value_buffer(2000000);
    std::atomic<bool> stop_flag(false);

    // 3. Запускаем потоки
    std::vector<std::thread> threads;
    std::cout << "Starting " << NUM_THREADS << " C++ worker threads..." << std::endl;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker_thread_loop, &policy_buffer, &value_buffer, &stop_flag);
    }

    // 4. Главный поток мониторит прогресс
    uint64_t last_head = 0;
    auto last_report_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_DURATION_SECONDS / 5; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_report_time).count();
        last_report_time = now;

        uint64_t current_head = policy_buffer.get_head();
        uint64_t samples_generated = current_head - last_head;
        last_head = current_head;
        double throughput = samples_generated / duration;

        std::cout << "\n--- STATS UPDATE ---" << std::endl;
        std::cout << "Time elapsed: " << (i + 1) * 5 << "s" << std::endl;
        std::cout << "Throughput: " << throughput << " traversals/sec" << std::endl;
        std::cout << "Buffer Fill -> Policy: " << policy_buffer.get_count() 
                  << " | Value: " << value_buffer.get_count() << std::endl;
        std::cout << "--------------------" << std::endl;
    }

    // 5. Останавливаем и ждем потоки
    std::cout << "Stopping threads..." << std::endl;
    stop_flag = true;
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    std::cout << "--- Test Finished ---" << std::endl;

    return 0;
}
