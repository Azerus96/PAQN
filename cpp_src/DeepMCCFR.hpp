#pragma once

#include <cstddef>
#include <vector>
#include <map>
#include <memory>
#include <random>
#include <atomic>
#include <functional> // <-- Добавим functional

#include <pybind11/pybind11.h>
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include "InferenceQueue.hpp"

namespace py = pybind11;

namespace ofc {

// --- ИЗМЕНЕНИЕ: Определяем типы для коллбэков ---
using RequestCallback = std::function<void(uint64_t, bool, const std::vector<float>&, const std::vector<std::vector<float>>&)>;
using ResultCallback = std::function<py::tuple(uint64_t)>;

class DeepMCCFR {
public:
    // --- ИЗМЕНЕНИЕ: Конструктор теперь принимает коллбэки ---
    DeepMCCFR(size_t action_limit,
              SharedReplayBuffer* policy_buffer, 
              SharedReplayBuffer* value_buffer, 
              RequestCallback request_cb,
              ResultCallback result_cb);
    
    void run_traversal();

private:
    HandEvaluator evaluator_;
    SharedReplayBuffer* policy_buffer_;
    SharedReplayBuffer* value_buffer_;
    
    // --- ИЗМЕНЕНИЕ: Храним коллбэки ---
    RequestCallback request_cb_;
    ResultCallback result_cb_;

    size_t action_limit_;
    std::mt19937 rng_;
    std::vector<float> dummy_action_vec_;
    
    static std::atomic<uint64_t> traversal_counter_;

    std::map<int, float> traverse(GameState& state, int traversing_player, bool is_root, uint64_t traversal_id);
    std::vector<float> featurize_state_cpp(const GameState& state, int player_view);
};

}
