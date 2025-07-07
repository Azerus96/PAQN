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

// ... (вспомогательные функции и конструктор без изменений) ...
std::vector<float> action_to_vector(const Action& action);
void add_dirichlet_noise(std::vector<float>& strategy, float alpha, std::mt19937& rng);
std::vector<float> action_to_vector(const Action& action) {
    std::vector<float> vec(ACTION_VECTOR_SIZE, 0.0f);
    const auto& placements = action.first;
    const auto& discarded_card = action.second;
    for (const auto& p : placements) {
        const auto& card = p.first;
        const auto& row_name = p.second.first;
        int slot_idx = -1;
        if (row_name == "top") slot_idx = 0;
        else if (row_name == "middle") slot_idx = 1;
        else if (row_name == "bottom") slot_idx = 2;
        if (slot_idx != -1 && card != INVALID_CARD) {
            vec[card * 4 + slot_idx] = 1.0f;
        }
    }
    if (discarded_card != INVALID_CARD) {
        vec[discarded_card * 4 + 3] = 1.0f;
    }
    return vec;
}
void add_dirichlet_noise(std::vector<float>& strategy, float alpha, std::mt19937& rng) {
    if (strategy.empty()) { return; }
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(strategy.size());
    float noise_sum = 0.0f;
    for (size_t i = 0; i < strategy.size(); ++i) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }
    if (noise_sum > 1e-6) {
        const float exploration_fraction = 0.25f;
        for (size_t i = 0; i < strategy.size(); ++i) {
            strategy[i] = (1.0f - exploration_fraction) * strategy[i] + exploration_fraction * (noise[i] / noise_sum);
        }
    }
}
std::atomic<uint64_t> DeepMCCFR::traversal_counter_{0};
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
std::vector<float> DeepMCCFR::featurize(const GameState& state, int player_view) {
    const Board& my_board = state.get_player_board(player_view);
    const Board& opp_board = state.get_opponent_board(player_view);
    std::vector<float> features(INFOSET_SIZE, 0.0f);
    int offset = 0;
    features[offset++] = static_cast<float>(state.get_street());
    features[offset++] = static_cast<float>(state.get_dealer_pos());
    features[offset++] = static_cast<float>(state.get_current_player());
    const auto& dealt_cards = state.get_dealt_cards();
    for (Card c : dealt_cards) {
        if (c != INVALID_CARD) features[offset + c] = 1.0f;
    }
    offset += 52;
    auto process_board = [&](const Board& board, int& current_offset) {
        for(int i=0; i<3; ++i) {
            if (board.top[i] != INVALID_CARD) features[current_offset + board.top[i]] = 1.0f;
        }
        current_offset += 52;
        for(int i=0; i<5; ++i) {
            if (board.middle[i] != INVALID_CARD) features[current_offset + board.middle[i]] = 1.0f;
        }
        current_offset += 52;
        for(int i=0; i<5; ++i) {
            if (board.bottom[i] != INVALID_CARD) features[current_offset + board.bottom[i]] = 1.0f;
        }
        current_offset += 52;
    };
    process_board(my_board, offset);
    process_board(opp_board, offset);
    const auto& my_discards = state.get_my_discards(player_view);
    for (Card c : my_discards) {
        if (c != INVALID_CARD) features[offset + c] = 1.0f;
    }
    offset += 52;
    features[offset++] = static_cast<float>(state.get_opponent_discard_count(player_view));
    features[offset++] = 0.0f;
    features[offset++] = 0.0f;
    return features;
}


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

    // <<< ИЗМЕНЕНИЕ: Упрощаем и делаем логику канонизации более надежной >>>
    
    // 1. Сначала создаем каноническое состояние и его инфосет
    std::map<int, int> suit_map_for_state;
    GameState canonical_state = state.get_canonical({}, suit_map_for_state); // Передаем пустой вектор действий
    std::vector<float> infoset_vec = featurize(canonical_state, traversing_player);

    // 2. Теперь строим канонические векторы действий, динамически создавая карту мастей
    std::vector<std::vector<float>> canonical_action_vectors;
    canonical_action_vectors.reserve(num_actions);
    
    std::map<int, int> suit_map_for_actions;
    int next_canonical_suit = 0;

    auto remap_card_safely = [&](Card& card) {
        if (card == INVALID_CARD) return;
        int original_suit = get_suit(card);
        // Если масть еще не встречалась, назначаем ей следующий свободный канонический номер
        if (suit_map_for_actions.find(original_suit) == suit_map_for_actions.end()) {
            suit_map_for_actions[original_suit] = next_canonical_suit++;
        }
        card = get_rank(card) * 4 + suit_map_for_actions[original_suit];
    };

    for (const auto& original_action : legal_actions) {
        Action canonical_action = original_action;
        // Сбрасываем карту мастей для каждого действия, чтобы обеспечить локальную инвариантность
        suit_map_for_actions.clear();
        next_canonical_suit = 0;
        // Применяем безопасное переназначение
        for (auto& placement : canonical_action.first) {
            remap_card_safely(placement.first);
        }
        remap_card_safely(canonical_action.second);
        canonical_action_vectors.push_back(action_to_vector(canonical_action));
    }
    
    // <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

    uint64_t policy_request_id = traversal_id * 2;
    {
        py::gil_scoped_acquire acquire;
        py::tuple request_tuple = py::make_tuple(
            policy_request_id, true, py::cast(infoset_vec), py::cast(canonical_action_vectors)
        );
        request_queue_->attr("put")(request_tuple);
    }

    std::vector<float> logits;
    {
        py::gil_scoped_acquire acquire;
        py::object result_obj = result_queue_->attr("get")(); 
        py::tuple result_tuple = result_obj.cast<py::tuple>();
        if (result_tuple[0].cast<uint64_t>() != policy_request_id) return {};
        logits = result_tuple[2].cast<std::vector<float>>();
    }

    std::vector<float> strategy(num_actions);
    if (!logits.empty() && logits.size() == num_actions) {
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
        
        value_buffer_->push(infoset_vec, dummy_action_vec_, action_payoffs[i].at(current_player));
        
        for(auto const& [player_idx, payoff] : action_payoffs[i]) {
            node_payoffs[player_idx] += strategy[i] * payoff;
        }
    }
    
    uint64_t value_request_id = traversal_id * 2 + 1;
    {
        py::gil_scoped_acquire acquire;
        py::tuple request_tuple = py::make_tuple(
            value_request_id, false, py::cast(infoset_vec), py::none()
        );
        request_queue_->attr("put")(request_tuple);
    }

    float value_baseline = 0.0f;
    {
        py::gil_scoped_acquire acquire;
        py::object result_obj = result_queue_->attr("get")();
        py::tuple result_tuple = result_obj.cast<py::tuple>();
        if (result_tuple[0].cast<uint64_t>() != value_request_id) return {};
        std::vector<float> predictions = result_tuple[2].cast<std::vector<float>>();
        if (predictions.empty()) return {};
        value_baseline = predictions[0];
    }

    for (int i = 0; i < num_actions; ++i) {
        float advantage = action_payoffs[i].at(current_player) - value_baseline;
        policy_buffer_->push(infoset_vec, canonical_action_vectors[i], advantage);
    }

    return node_payoffs;
}

}
