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

// ... (action_to_vector, add_dirichlet_noise, конструктор, run_traversal, featurize_state_cpp - без изменений) ...
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

std::vector<float> DeepMCCFR::featurize_state_cpp(const GameState& state, int player_view) {
    std::vector<float> features(INFOSET_SIZE, 0.0f);
    
    const int P_BOARD_TOP = 0, P_BOARD_MID = 1, P_BOARD_BOT = 2, P_HAND = 3;
    const int O_BOARD_TOP = 4, O_BOARD_MID = 5, O_BOARD_BOT = 6;
    const int P_DISCARDS = 7, DECK_REMAINING = 8;
    const int IS_STREET_1 = 9, IS_STREET_2 = 10, IS_STREET_3 = 11, IS_STREET_4 = 12, IS_STREET_5 = 13;
    const int O_DISCARD_COUNT = 14, TURN = 15;
    
    const int plane_size = NUM_SUITS * NUM_RANKS;

    auto set_card = [&](int channel, Card card) {
        if (card != INVALID_CARD) {
            int suit = get_suit(card);
            int rank = get_rank(card);
            features[channel * plane_size + suit * NUM_RANKS + rank] = 1.0f;
        }
    };

    const Board& my_board = state.get_player_board(player_view);
    const Board& opp_board = state.get_opponent_board(player_view);

    for (Card c : my_board.top) set_card(P_BOARD_TOP, c);
    for (Card c : my_board.middle) set_card(P_BOARD_MID, c);
    for (Card c : my_board.bottom) set_card(P_BOARD_BOT, c);
    
    for (Card c : state.get_dealt_cards()) set_card(P_HAND, c);

    for (Card c : opp_board.top) set_card(O_BOARD_TOP, c);
    for (Card c : opp_board.middle) set_card(O_BOARD_MID, c);
    for (Card c : opp_board.bottom) set_card(O_BOARD_BOT, c);

    for (Card c : state.get_my_discards(player_view)) set_card(P_DISCARDS, c);

    std::vector<bool> known_cards(52, false);
    auto mark_known = [&](Card c) { if (c != INVALID_CARD) known_cards[c] = true; };
    for (Card c : my_board.get_all_cards()) mark_known(c);
    for (Card c : opp_board.get_all_cards()) mark_known(c);
    for (Card c : state.get_dealt_cards()) mark_known(c);
    for (Card c : state.get_my_discards(player_view)) mark_known(c);
    
    for (int c = 0; c < 52; ++c) {
        if (!known_cards[c]) {
            set_card(DECK_REMAINING, c);
        }
    }

    int street = state.get_street();
    if (street >= 1 && street <= 5) {
        int street_channel = IS_STREET_1 + (street - 1);
        std::fill(features.begin() + street_channel * plane_size, features.begin() + (street_channel + 1) * plane_size, 1.0f);
    }

    float opp_discard_val = static_cast<float>(state.get_opponent_discard_count(player_view)) / 4.0f;
    std::fill(features.begin() + O_DISCARD_COUNT * plane_size, features.begin() + (O_DISCARD_COUNT + 1) * plane_size, opp_discard_val);

    if (state.get_current_player() == player_view) {
        std::fill(features.begin() + TURN * plane_size, features.begin() + (TURN + 1) * plane_size, 1.0f);
    }

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

    std::map<int, int> suit_map;
    GameState canonical_state = state.get_canonical(suit_map);
    std::vector<float> infoset_vec = featurize_state_cpp(canonical_state, current_player);
    
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
    uint64_t value_request_id = traversal_id * 2 + 1;
    bool is_traversing_players_turn = (current_player == traversing_player);

    {
        py::gil_scoped_acquire acquire;
        py::tuple policy_request_tuple = py::make_tuple(
            policy_request_id, true, py::cast(infoset_vec), py::cast(canonical_action_vectors)
        );
        request_queue_->attr("put")(policy_request_tuple);

        if (is_traversing_players_turn) {
            py::tuple value_request_tuple = py::make_tuple(
                value_request_id, false, py::cast(infoset_vec), py::none()
            );
            request_queue_->attr("put")(value_request_tuple);
        }
    }

    std::vector<float> logits;
    float value_baseline = 0.0f;
    int results_to_get = is_traversing_players_turn ? 2 : 1;
    int results_gotten = 0;

    while(results_gotten < results_to_get) {
        bool got_item = false;
        {
            py::gil_scoped_acquire acquire;
            auto queue_module = py::module_::import("queue");
            auto PyExc_Empty = queue_module.attr("Empty");
            
            try {
                py::object result_obj = result_queue_->attr("get_nowait")();
                
                // Если мы здесь, значит, получили объект. Обработаем его внутри GIL.
                got_item = true;
                py::tuple result_tuple = result_obj.cast<py::tuple>();
                uint64_t result_id = result_tuple[0].cast<uint64_t>();
                
                if (result_id == policy_request_id) {
                    logits = result_tuple[2].cast<std::vector<float>>();
                    results_gotten++;
                } else if (result_id == value_request_id) {
                    std::vector<float> predictions = result_tuple[2].cast<std::vector<float>>();
                    if (!predictions.empty()) {
                        value_baseline = predictions[0];
                    }
                    results_gotten++;
                } else {
                    // Не наш результат, вернем в очередь
                    result_queue_->attr("put")(result_obj);
                }
            } catch (const py::error_already_set& e) {
                if (!e.matches(PyExc_Empty)) {
                    // Другая ошибка, пробрасываем ее
                    throw;
                }
                // Если очередь пуста (e.matches(PyExc_Empty)), ничего не делаем, got_item останется false
            }
        } // GIL отпускается здесь

        if (!got_item) {
            // Если очередь была пуста, немного поспим, чтобы не грузить CPU
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
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

    if (is_root && is_traversing_players_turn) {
        add_dirichlet_noise(strategy, 0.3f, rng_);
    }

    std::discrete_distribution<int> dist(strategy.begin(), strategy.end());
    int sampled_action_idx = dist(rng_);

    if (is_traversing_players_turn) {
        state.apply_action(legal_actions[sampled_action_idx], traversing_player, undo_info);
        auto action_payoffs = traverse(state, traversing_player, false, traversal_id);
        state.undo_action(undo_info, traversing_player);

        auto it = action_payoffs.find(current_player);
        if (it != action_payoffs.end()) {
            float advantage = it->second - value_baseline;
            policy_buffer_->push(infoset_vec, canonical_action_vectors[sampled_action_idx], advantage);
            value_buffer_->push(infoset_vec, dummy_action_vec_, it->second);
        }
        return action_payoffs;

    } else { // Ход оппонента
        state.apply_action(legal_actions[sampled_action_idx], traversing_player, undo_info);
        auto result = traverse(state, traversing_player, false, traversal_id);
        state.undo_action(undo_info, traversing_player);
        return result;
    }
}

} // namespace ofc
