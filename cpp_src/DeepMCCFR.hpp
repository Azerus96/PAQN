#pragma once
#include <pybind11/pybind11.h>
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include "InferenceQueue.hpp"
#include <vector>
#include <map>
#include <memory>
#include <random>
#include <atomic>

namespace py = pybind11;

namespace ofc {

class DeepMCCFR {
public:
    // --- ИЗМЕНЕНИЕ: Вернули action_limit ---
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
    
    InferenceRequestQueue* request_queue_;
    InferenceResultQueue* result_queue_;

    // --- ИЗМЕНЕНИЕ: Вернули action_limit_ ---
    size_t action_limit_;
    std::mt19937 rng_;
    std::vector<float> dummy_action_vec_;
    
    static std::atomic<uint64_t> traversal_counter_;

    std::map<int, float> traverse(GameState& state, int traversing_player, bool is_root, uint64_t traversal_id);
    std::vector<int> serialize_state(const GameState& state, int player_view);
};

}
