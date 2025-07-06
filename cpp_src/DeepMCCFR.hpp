#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include "InferenceQueue.hpp"
#include <vector>
#include <map>
#include <memory>
#include <random>

namespace ofc {

class DeepMCCFR {
public:
    // Конструктор теперь принимает два указателя на буферы
    DeepMCCFR(size_t action_limit, SharedReplayBuffer* policy_buffer, SharedReplayBuffer* value_buffer, InferenceQueue* queue);
    
    void run_traversal();

private:
    HandEvaluator evaluator_;
    SharedReplayBuffer* policy_buffer_; // Буфер для (s, a, advantage)
    SharedReplayBuffer* value_buffer_;  // Буфер для (s, value)
    InferenceQueue* inference_queue_;
    size_t action_limit_;
    std::mt19937 rng_;
    std::vector<float> dummy_action_vec_; // Фиктивный вектор для value_buffer

    std::map<int, float> traverse(GameState& state, int traversing_player, bool is_root);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
