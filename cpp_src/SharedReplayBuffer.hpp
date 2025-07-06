#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <mutex>
#include "constants.hpp"
#include <chrono> // Добавлено для неблокирующего сида

namespace ofc {

// ИЗМЕНЕНО: Структура теперь хранит один пример (действие)
struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> action_vector;
    float target_regret;

    TrainingSample() 
        : infoset_vector(INFOSET_SIZE), 
          action_vector(ACTION_VECTOR_SIZE, 0.0f), 
          target_regret(0.0f) {}
};

class SharedReplayBuffer {
public:
    // ИЗМЕНЕНО: Конструктор больше не принимает action_limit
    SharedReplayBuffer(uint64_t capacity) 
        : capacity_(capacity), head_(0), count_(0)
    {
        buffer_.resize(capacity_); // Просто создаем пустые сэмплы
        // ИЗМЕНЕНО: Используем неблокирующий сид на основе времени, чтобы избежать блокировки
        rng_.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::cout << "C++: Parametric Replay Buffer created with capacity " << capacity << std::endl;
    }

    // ИЗМЕНЕНО: push теперь принимает один пример за раз
    void push(const std::vector<float>& infoset_vec, const std::vector<float>& action_vec, float regret) {
        std::lock_guard<std::mutex> lock(mtx_);
        uint64_t index = head_ % capacity_;
        head_++;

        auto& sample = buffer_[index];
        std::copy(infoset_vec.begin(), infoset_vec.end(), sample.infoset_vector.begin());
        std::copy(action_vec.begin(), action_vec.end(), sample.action_vector.begin());
        sample.target_regret = regret;
        
        if (count_ < capacity_) {
            count_++;
        }
    }

    // ИЗМЕНЕНО: sample теперь возвращает три указателя (инфосеты, действия, сожаления)
    void sample(int batch_size, float* out_infosets, float* out_actions, float* out_regrets) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        if (count_ < static_cast<uint64_t>(batch_size)) {
            std::fill(out_infosets, out_infosets + batch_size * INFOSET_SIZE, 0.0f);
            std::fill(out_actions, out_actions + batch_size * ACTION_VECTOR_SIZE, 0.0f);
            std::fill(out_regrets, out_regrets + batch_size, 0.0f);
            return;
        }

        std::uniform_int_distribution<uint64_t> dist(0, count_ - 1);

        for (int i = 0; i < batch_size; ++i) {
            uint64_t sample_idx = dist(rng_);
            const auto& sample = buffer_[sample_idx];
            std::copy(sample.infoset_vector.begin(), sample.infoset_vector.end(), out_infosets + i * INFOSET_SIZE);
            std::copy(sample.action_vector.begin(), sample.action_vector.end(), out_actions + i * ACTION_VECTOR_SIZE);
            *(out_regrets + i) = sample.target_regret;
        }
    }
    
    uint64_t get_count() {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }

    uint64_t get_head() {
        std::lock_guard<std::mutex> lock(mtx_);
        return head_;
    }
    
private:
    std::vector<TrainingSample> buffer_;
    uint64_t capacity_;
    uint64_t head_;
    uint64_t count_;
    std::mutex mtx_;
    std::mt19937 rng_;
};

}
