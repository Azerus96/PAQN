#pragma once
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>

// Запрос на инференс, который C++ отправляет в Python
struct InferenceRequest {
    std::vector<float> infoset;
    std::promise<std::vector<float>> promise;
    int num_actions;
};

// Потокобезопасная очередь для запросов
class InferenceQueue {
public:
    void push(InferenceRequest&& request) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push_back(std::move(request));
        lock.unlock();
        cv_.notify_one();
    }

    // Забирает все запросы из очереди, не блокируя
    std::vector<InferenceRequest> pop_all() {
        std::vector<InferenceRequest> requests;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            if (!queue_.empty()) {
                requests.reserve(queue_.size());
                std::move(queue_.begin(), queue_.end(), std::back_inserter(requests));
                queue_.clear();
            }
        }
        return requests;
    }

    // Ждет, пока в очереди не появится хотя бы один элемент
    void wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
    }

private:
    std::deque<InferenceRequest> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
};
