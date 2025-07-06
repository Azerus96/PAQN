#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/functional.h> // Для std::function
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include <vector>
#include <map>
#include <memory>
#include <random>
#include <functional> // Явно добавим для std::function

namespace py = pybind11;

namespace ofc {

// Тип для Python-функции, которую мы будем вызывать для Policy-инференса
// Принимает: инфосет, векторы действий, функцию для ответа
using PolicyInferenceCallback = std::function<
    void(
        const std::vector<float>&, 
        const std::vector<std::vector<float>>&,
        const std::function<void(std::vector<float>)>&
    )
>;

// Тип для Python-функции, которую мы будем вызывать для Value-инференса
// Принимает: инфосет, функцию для ответа
using ValueInferenceCallback = std::function<
    void(
        const std::vector<float>&,
        const std::function<void(float)>&
    )
>;


class DeepMCCFR {
public:
    // Конструктор теперь принимает callback-функции
    DeepMCCFR(size_t action_limit, SharedReplayBuffer* policy_buffer, SharedReplayBuffer* value_buffer, 
              PolicyInferenceCallback policy_cb, ValueInferenceCallback value_cb);
    
    void run_traversal();

private:
    HandEvaluator evaluator_;
    SharedReplayBuffer* policy_buffer_;
    SharedReplayBuffer* value_buffer_;
    PolicyInferenceCallback policy_callback_;
    ValueInferenceCallback value_callback_;
    size_t action_limit_;
    std::mt19937 rng_;
    std::vector<float> dummy_action_vec_;

    std::map<int, float> traverse(GameState& state, int traversing_player, bool is_root);
    std::vector<float> featurize(const GameState& state, int player_view);
};

} // namespace ofc
