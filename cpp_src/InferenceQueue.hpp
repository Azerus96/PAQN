// --- START OF FILE PAQN-main/cpp_src/InferenceQueue.hpp ---
#pragma once
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <variant> // ДОБАВЛЕНО: Для использования std::variant
#include "constants.hpp"

namespace ofc {

// УДАЛЕНО: enum RequestType больше не нужен, так как std::variant сам хранит тип.

// Данные для запроса к Policy Network
struct PolicyRequestData {
    std::vector<float> infoset;
    std::vector<std::vector<float>> action_vectors;
};

// Данные для запроса к Value Network
struct ValueRequestData {
    std::vector<float> infoset;
};

// Универсальный запрос на инференс
struct InferenceRequest {
    // ИЗМЕНЕНО: union заменен на std::variant. Это типобезопасно и решает ошибку компиляции.
    std::variant<PolicyRequestData, ValueRequestData> data;

    // Promise будет возвращать вектор float в обоих случаях
    std::promise<std::vector<float>> promise;

    // УДАЛЕНО: Ручное управление конструкторами, деструкторами и операторами перемещения/копирования
    // больше не требуется. std::variant делает все это автоматически и безопасно.
};

// Потокобезопасная очередь для запросов (без изменений)
class InferenceQueue {
public:
    void push(InferenceRequest&& request) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push_back(std::move(request));
        lock.unlock();
        cv_.notify_one();
    }

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

    void wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
    }

private:
    std::deque<InferenceRequest> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace ofc
// --- END OF FILE PAQN-main/cpp_src/InferenceQueue.hpp ---
