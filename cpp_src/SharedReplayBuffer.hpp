#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <mutex>
#include "constants.hpp"
#include <chrono>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace ofc {

struct TrainingSample {
    std::vector<float> infoset;
    std::vector<float> action_vector;
    float target_value;

    TrainingSample() 
        : infoset(INFOSET_SIZE, 0.0f), 
          action_vector(ACTION_VECTOR_SIZE, 0.0f), 
          target_value(0.0f) {}
};

class SharedReplayBuffer {
public:
    SharedReplayBuffer(uint64_t capacity) 
        : capacity_(capacity), head_(0), count_(0)
    {
        buffer_.resize(capacity_);
        thread_local static std::mt19937 rng(
            static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + 
            static_cast<unsigned int>(std::hash<std::thread::id>{}(std::this_thread::get_id()))
        );
        rng_ = &rng;
        std::cout << "C++: Replay Buffer created with capacity " << capacity << std::endl;
    }

    void push(const std::vector<float>& infoset_vec, const std::vector<float>& action_vec, float target) {
        std::lock_guard<std::mutex> lock(mtx_);
        uint64_t index = head_ % capacity_;
        head_++;

        auto& sample = buffer_[index];
        std::copy(infoset_vec.begin(), infoset_vec.end(), sample.infoset.begin());
        std::copy(action_vec.begin(), action_vec.end(), sample.action_vector.begin());
        sample.target_value = target;
        
        if (count_ < capacity_) {
            count_++;
        }
    }

    bool sample(int batch_size, py::array_t<float>& out_infosets, py::array_t<float>& out_actions, py::array_t<float>& out_targets) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        if (count_ < static_cast<uint64_t>(batch_size)) {
            return false;
        }

        std::uniform_int_distribution<uint64_t> dist(0, count_ - 1);
        
        auto infosets_ptr = static_cast<float*>(out_infosets.request().ptr);
        auto actions_ptr = static_cast<float*>(out_actions.request().ptr);
        auto targets_ptr = static_cast<float*>(out_targets.request().ptr);

        for (int i = 0; i < batch_size; ++i) {
            uint64_t sample_idx = dist(*rng_);
            const auto& sample = buffer_[sample_idx];
            
            std::copy(sample.infoset.begin(), sample.infoset.end(), infosets_ptr + i * INFOSET_SIZE);
            std::copy(sample.action_vector.begin(), sample.action_vector.end(), actions_ptr + i * ACTION_VECTOR_SIZE);
            *(targets_ptr + i) = sample.target_value;
        }
        return true;
    }
    
    uint64_t size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }

    uint64_t total_generated() {
        std::lock_guard<std::mutex> lock(mtx_);
        return head_;
    }
    
private:
    std::vector<TrainingSample> buffer_;
    uint64_t capacity_;
    uint64_t head_;
    uint64_t count_;
    std::mutex mtx_;
    std::mt19937* rng_;
};

} // namespace ofc
