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

std::vector<int> DeepMCCFR::serialize_state(const GameState& state, int player_view) {
    std::vector<int> raw_state;
    raw_state.reserve(100);

    const Board& my_board = state.get_player_board(player_view);
    const Board& opp_board = state.get_opponent_board(player_view);

    raw_state.push_back(state.get_street());
    raw_state.push_back(state.get_current_player());
    raw_state.push_back(player_view);

    const auto& hand = state.get_dealt_cards();
    raw_state.push_back(hand.size());
    for (Card c : hand) raw_state.push_back(c);

    for (Card c : my_board.top) raw_state.push_back(c);
    for (Card c : my_board.middle) raw_state.push_back(c);
    for (Card c : my_board.bottom) raw_state.push_back(c);

    for (Card c : opp_board.top) raw_state.push_back(c);
    for (Card c : opp_board.middle) raw_state.push_back(c);
    for (Card c : opp_board.bottom) raw_state.push_back(c);
    
    raw_state.push_back(0); // is_player_fantasyland
    raw_state.push_back(0); // is_opponent_fantasyland

    return raw_state;
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

    std::map<int, int> suit_map;
    GameState canonical_state = state.get_canonical(suit_map);
    std::vector<int> raw_state_vec = serialize_state(canonical_state, current_player);
    
    std::vector<std::vector<float>> canonical_action_vectors;
    canonical_action_vectors.reserve(num_actions);
    
    auto remap_card_safely = [&](Card& card) {
        if (card == INVALID_CARD) return;
        auto it = suit_map.find(get_suit(card));
        if (it != suit_map.end()) {
            card = get_rank(card) * 4 + it->second;
        } else {
            card = INVALID_CARD; 
        }
    };

    for (const auto& original_action : legal_actions) {
        Action canonical_action = original_action;
        for (auto& placement : canonical_action.first) {
            remap_card_safely(placement.first);
        }
        remap_card_safely(canonical_action.second);
        canonical_action_vectors.push_back(action_to_vector(canonical_action));
    }
    
    uint64_t policy_request_id = traversal_id * 2;
    {
        py::gil_scoped_acquire acquire;
        py::tuple request_tuple = py::make_tuple(
            policy_request_id, true, py::cast(raw_state_vec), py::cast(canonical_action_vectors)
        );
        request_queue_->attr("put")(request_tuple);
    }

    std::vector<float> logits;
    {
        py::gil_scoped_acquire acquire;
        py::object result_obj = result_queue_->attr("get")(); 
        py::tuple result_tuple = result_obj.cast<py::tuple>();
        if (result_tuple[0].cast<uint64_t>() != policy_request_id) {
            std::cerr << "C++ Thread " << std::this_thread::get_id() << ": Policy ID mismatch! Expected " << policy_request_id << ", got " << result_tuple[0].cast<uint64_t>() << std::endl;
            return {};
        }
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

    if (is_root && current_player == traversing_player) {
        add_dirichlet_noise(strategy, 0.3f, rng_);
    }

    std::discrete_distribution<int> dist(strategy.begin(), strategy.end());
    int sampled_action_idx = dist(rng_);

    if (current_player == traversing_player) {
        uint64_t value_request_id = traversal_id * 2 + 1;
        {
            py::gil_scoped_acquire acquire;
            py::tuple request_tuple = py::make_tuple(
                value_request_id, false, py::cast(raw_state_vec), py::none()
            );
            request_queue_->attr("put")(request_tuple);
        }

        float value_baseline = 0.0f;
        {
            py::gil_scoped_acquire acquire;
            py::object result_obj = result_queue_->attr("get")();
            py::tuple result_tuple = result_obj.cast<py::tuple>();
            if (result_tuple[0].cast<uint64_t>() == value_request_id) {
                std::vector<float> predictions = result_tuple[2].cast<std::vector<float>>();
                if (!predictions.empty()) {
                    value_baseline = predictions[0];
                }
            }
        }

        state.apply_action(legal_actions[sampled_action_idx], traversing_player, undo_info);
        auto action_payoffs = traverse(state, traversing_player, false, traversal_id);
        state.undo_action(undo_info, traversing_player);

        auto it = action_payoffs.find(current_player);
        if (it != action_payoffs.end()) {
            float advantage = it->second - value_baseline;
            policy_buffer_->push_raw(raw_state_vec, canonical_action_vectors[sampled_action_idx], advantage);
            value_buffer_->push_raw(raw_state_vec, dummy_action_vec_, it->second);
        }
        return action_payoffs;

    } else {
        state.apply_action(legal_actions[sampled_action_idx], traversing_player, undo_info);
        auto result = traverse(state, traversing_player, false, traversal_id);
        state.undo_action(undo_info, traversing_player);
        return result;
    }
}

} // namespace ofc
