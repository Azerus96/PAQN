#include "DeepMCCFR.hpp"
#include "constants.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>

namespace ofc {

// ... (вспомогательные функции action_to_vector и add_dirichlet_noise без изменений) ...
// ...

std::atomic<uint64_t> DeepMCCFR::traversal_counter_{0};

// <<< ИЗМЕНЕНИЕ: Конструктор теперь принимает очереди
DeepMCCFR::DeepMCCFR(size_t action_limit, SharedReplayBuffer* policy_buffer, SharedReplayBuffer* value_buffer,
                     InferenceRequestQueue* request_queue, InferenceResultQueue* result_queue) 
    : action_limit_(action_limit), 
      policy_buffer_(policy_buffer), 
      value_buffer_(value_buffer),
      request_queue_(request_queue),
      result_queue_(result_queue),
      rng_(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + 
           static_cast<unsigned int>(std::hash<std::thread::id>{}(std::this_thread::get_id()))),
      dummy_action_vec_(ACTION_VECTOR_SIZE, 0.0f)
{}

void DeepMCCFR::run_traversal() {
    uint64_t traversal_id = ++traversal_counter_;
    GameState state; 
    traverse(state, 0, true, traversal_id);
    state.reset(); 
    traverse(state, 1, true, traversal_id);
}

// ... (featurize без изменений) ...
// ...

std::map<int, float> DeepMCCFR::traverse(GameState& state, int traversing_player, bool is_root, uint64_t traversal_id) {
    if (state.is_terminal()) {
        auto payoffs = state.get_payoffs(evaluator_);
        return {{0, payoffs.first}, {1, payoffs.second}};
    }

    int current_player = state.get_current_player();
    
    std::vector<Action> legal_actions;
    state.get_legal_actions(action_limit_, legal_actions, rng_);
    int num_actions = legal_actions.size();
    UndoInfo undo_info;

    if (num_actions == 0) {
        state.apply_action({{}, INVALID_CARD}, traversing_player, undo_info);
        auto result = traverse(state, traversing_player, false, traversal_id);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    if (current_player != traversing_player) {
        int action_idx = std::uniform_int_distribution<int>(0, num_actions - 1)(rng_);
        state.apply_action(legal_actions[action_idx], traversing_player, undo_info);
        auto result = traverse(state, traversing_player, false, traversal_id);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    std::map<int, int> suit_map;
    GameState canonical_state = state.get_canonical(suit_map);
    std::vector<float> infoset_vec = featurize(canonical_state, traversing_player);
    
    std::vector<std::vector<float>> canonical_action_vectors;
    canonical_action_vectors.reserve(num_actions);
    auto remap_card = [&](Card& card) {
        if (card == INVALID_CARD) return;
        card = get_rank(card) * 4 + suit_map.at(get_suit(card));
    };
    for (const auto& original_action : legal_actions) {
        Action canonical_action = original_action;
        for (auto& placement : canonical_action.first) remap_card(placement.first);
        remap_card(canonical_action.second);
        canonical_action_vectors.push_back(action_to_vector(canonical_action));
    }

    // <<< ИЗМЕНЕНИЕ: Логика инференса полностью переписана. >>>
    // 1. Отправляем запрос в очередь
    uint64_t policy_request_id = traversal_id * 2;
    {
        py::gil_scoped_acquire acquire;
        request_queue_->attr("put")(InferenceRequest{policy_request_id, true, infoset_vec, canonical_action_vectors});
    }

    // 2. Получаем результат из другой очереди
    std::vector<float> logits;
    {
        py::gil_scoped_acquire acquire;
        // Блокируемся на Python-очереди, GIL будет отпущен во время ожидания
        py::object result_obj = result_queue_->attr("get")(); 
        InferenceResult result = result_obj.cast<InferenceResult>();
        // Простая проверка, что мы получили то, что ждали. В проде можно убрать.
        if (result.id != policy_request_id || !result.is_policy_result) {
            // Обработка ошибки - например, возврат пустого результата
            return {};
        }
        logits = std::move(result.predictions);
    }
    // <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

    std::vector<float> strategy(num_actions);
    if (!logits.empty()) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for(float l : logits) if(l > max_logit) max_logit = l;
        float sum_exp = 0.0f;
        for (int i = 0; i < num_actions; ++i) {
            strategy[i] = std::exp(logits[i] - max_logit);
            sum_exp += strategy[i];
        }
        if (sum_exp > 1e-6) {
            for (int i = 0; i < num_actions; ++i) strategy[i] /= sum_exp;
        } else {
            std::fill(strategy.begin(), strategy.end(), 1.0f / num_actions);
        }
    } else {
        std::fill(strategy.begin(), strategy.end(), 1.0f / num_actions);
    }

    if (is_root) {
        add_dirichlet_noise(strategy, 0.3f, rng_);
    }

    std::vector<std::map<int, float>> action_payoffs(num_actions);
    std::map<int, float> node_payoffs = {{0, 0.0f}, {1, 0.0f}};

    for (int i = 0; i < num_actions; ++i) {
        state.apply_action(legal_actions[i], traversing_player, undo_info);
        action_payoffs[i] = traverse(state, traversing_player, false, traversal_id);
        state.undo_action(undo_info, traversing_player);
        for(auto const& [player_idx, payoff] : action_payoffs[i]) {
            node_payoffs[player_idx] += strategy[i] * payoff;
        }
    }
    
    // <<< ИЗМЕНЕНИЕ: Логика инференса для Value Network >>>
    uint64_t value_request_id = traversal_id * 2 + 1;
    {
        py::gil_scoped_acquire acquire;
        // Для value-запроса action_vectors пустые
        request_queue_->attr("put")(InferenceRequest{value_request_id, false, infoset_vec, {}});
    }

    float value_baseline = 0.0f;
    {
        py::gil_scoped_acquire acquire;
        py::object result_obj = result_queue_->attr("get")();
        InferenceResult result = result_obj.cast<InferenceResult>();
        if (result.id != value_request_id || result.is_policy_result || result.predictions.empty()) {
            return {};
        }
        value_baseline = result.predictions[0];
    }
    // <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

    // <<< ИЗМЕНЕНИЕ: Исправлена логическая ошибка.
    // Теперь мы сохраняем канонический инфосет и канонические векторы действий.
    // А таргеты (regret/value) считаются на основе оригинальных, не-канонических исходов.
    // Это правильно, т.к. модель должна учиться на канонических представлениях.
    value_buffer_->push(infoset_vec, dummy_action_vec_, node_payoffs.at(current_player));

    for (int i = 0; i < num_actions; ++i) {
        float advantage = action_payoffs[i].at(current_player) - value_baseline;
        policy_buffer_->push(infoset_vec, canonical_action_vectors[i], advantage);
    }

    return node_payoffs;
}

}
