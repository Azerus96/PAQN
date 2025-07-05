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
    // --- УДАЛЕНО: Константа EPSILON для ε-greedy больше не нужна. ---

    HandEvaluator evaluator_;
    SharedReplayBuffer* replay_buffer_; 
    InferenceQueue* inference_queue_;
    size_t action_limit_;
    std::mt19937 rng_;

    // --- ИЗМЕНЕНИЕ: Добавлен параметр is_root для отслеживания корневого узла обхода. ---
    // Это необходимо, чтобы применять шум Дирихле только один раз в начале обхода для каждого игрока.
    std::map<int, float> traverse(GameState& state, int traversing_player, bool is_root);
    
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
