#pragma once
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include "constants.hpp"

namespace ofc {

// Тип запроса к нейронной сети
enum class RequestType {
    POLICY,
    VALUE
};

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
    RequestType type;
    
    // Используем union, чтобы хранить данные только одного типа
    union Data {
        PolicyRequestData policy_data;
        ValueRequestData value_data;

        // Конструкторы и деструкторы для правильного управления union
        Data() {}
        ~Data() {}
    } data;

    // Promise будет возвращать вектор float в обоих случаях
    std::promise<std::vector<float>> promise;

    // Конструктор и деструктор для правильного управления union
    InferenceRequest() : type(RequestType::POLICY) {
        new (&data.policy_data) PolicyRequestData();
    }

    ~InferenceRequest() {
        if (type == RequestType::POLICY) {
            data.policy_data.~PolicyRequestData();
        } else {
            data.value_data.~ValueRequestData();
        }
    }

    // Запрещаем копирование, чтобы избежать проблем с union
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;

    // Разрешаем перемещение
    InferenceRequest(InferenceRequest&& other) noexcept : type(other.type), promise(std::move(other.promise)) {
        if (type == RequestType::POLICY) {
            new (&data.policy_data) PolicyRequestData(std::move(other.data.policy_data));
        } else {
            new (&data.value_data) ValueRequestData(std::move(other.data.value_data));
        }
    }
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
