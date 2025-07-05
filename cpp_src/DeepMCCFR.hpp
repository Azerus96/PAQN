#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include "InferenceQueue.hpp"
#include <vector>
#include <map>
#include <memory>
#include <random> // <-- ДОБАВЛЕНО

namespace ofc {

class DeepMCCFR {
public:
    DeepMCCFR(size_t action_limit, SharedReplayBuffer* buffer, InferenceQueue* queue);
    
    void run_traversal();

private:
    HandEvaluator evaluator_;
    SharedReplayBuffer* replay_buffer_; 
    InferenceQueue* inference_queue_;
    size_t action_limit_;
    std::mt19937 rng_; // <-- RNG теперь здесь, для каждого воркера свой

    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
