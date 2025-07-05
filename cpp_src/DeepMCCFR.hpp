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
    DeepMCCFR(size_t action_limit, SharedReplayBuffer* buffer, InferenceQueue* queue);
    
    void run_traversal();

private:
    // --- ИЗМЕНЕНИЕ: ДОБАВЛЕНА КОНСТАНТА EPSILON ---
    // Вероятность выбора случайного действия для исследования (10%)
    static constexpr double EPSILON = 0.25; 
    // --- КОНЕЦ ИЗМЕНЕНИЯ ---

    HandEvaluator evaluator_;
    SharedReplayBuffer* replay_buffer_; 
    InferenceQueue* inference_queue_;
    size_t action_limit_;
    std::mt19937 rng_;

    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
