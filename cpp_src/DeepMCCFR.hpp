#pragma once
#include <pybind11/pybind11.h>
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include "InferenceQueue.hpp" // <<< ИЗМЕНЕНИЕ: Подключаем новый заголовок для очередей
#include <vector>
#include <map>
#include <memory>
#include <random>
#include <atomic>

namespace py = pybind11;

namespace ofc {

// <<< ИЗМЕНЕНИЕ: Полностью убираем std::function коллбэки.
// Вместо них будем использовать очереди напрямую.

class DeepMCCFR {
public:
    // <<< ИЗМЕНЕНИЕ: Конструктор теперь принимает указатели на очереди.
    DeepMCCFR(size_t action_limit, 
              SharedReplayBuffer* policy_buffer, 
              SharedReplayBuffer* value_buffer, 
              InferenceRequestQueue* request_queue,
              InferenceResultQueue* result_queue);
    
    void run_traversal();

private:
    HandEvaluator evaluator_;
    SharedReplayBuffer* policy_buffer_;
    SharedReplayBuffer* value_buffer_;
    
    // <<< ИЗМЕНЕНИЕ: Храним указатели на очереди.
    InferenceRequestQueue* request_queue_;
    InferenceResultQueue* result_queue_;

    size_t action_limit_;
    std::mt19937 rng_;
    std::vector<float> dummy_action_vec_;
    
    static std::atomic<uint64_t> traversal_counter_;

    std::map<int, float> traverse(GameState& state, int traversing_player, bool is_root, uint64_t traversal_id);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
