#pragma once
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <variant>
#include <iostream> 
#include <thread>   
#include "constants.hpp"

namespace ofc {

struct PolicyRequestData {
    std::vector<float> infoset;
    std::vector<std::vector<float>> action_vectors;
};

struct ValueRequestData {
    std::vector<float> infoset;
};

struct InferenceRequest {
    std::variant<PolicyRequestData, ValueRequestData> data;
    std::promise<std::vector<float>> promise;
};

class InferenceQueue {
public:
    void push(InferenceRequest&& request) {
        std::unique_lock<std::mutex> lock(mtx_);
        std::cout << "[C++ Queue] push(): Before size=" << queue_.size() << std::endl << std::flush;
        queue_.push_back(std::move(request));
        std::cout << "[C++ Queue] push(): After size=" << queue_.size() << ". Notifying." << std::endl << std::flush;
        lock.unlock();
        cv_.notify_one();
    }

    std::vector<InferenceRequest> pop_all() {
        std::vector<InferenceRequest> requests;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            std::cout << "[C++ Queue] pop_all(): Popping " << queue_.size() << " requests." << std::endl << std::flush;
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
        std::cout << "[C++ Queue] wait(): Entering, queue empty? " << (queue_.empty() ? "Yes" : "No") << ". Waiting..." << std::endl << std::flush;
        cv_.wait(lock, [this] { return !queue_.empty(); });
        std::cout << "[C++ Queue] wait(): Woke up, queue size: " << queue_.size() << std::endl << std::flush;
    }

private:
    std::deque<InferenceRequest> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

}
