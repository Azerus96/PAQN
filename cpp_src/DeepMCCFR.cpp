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
#include <mutex>
#include <condition_variable>

namespace ofc {

// Инициализация статического счетчика
std::atomic<uint64_t> DeepMCCFR::traversal_counter_{0};

// Вспомогательные функции (без изменений)
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


// Измененный конструктор
DeepMCCFR::DeepMCCFR(size_t action_limit, SharedReplayBuffer* policy_buffer, SharedReplayBuffer* value_buffer,
                     PolicyInferenceCallback policy_cb, ValueInferenceCallback value_cb) 
    : action_limit_(action_limit), 
      policy_buffer_(policy_buffer), 
      value_buffer_(value_buffer),
      policy_callback_(std::move(policy_cb)),
      value_callback_(std::move(value_cb)),
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
        for(int i=0; i<3; ++i) features[current_offset + i*53 + (board.top[i] == INVALID_CARD ? 52 : board.top[i])] = 1.0f;
        current_offset += 3 * 53;
        for(int i=0; i<5; ++i) features[current_offset + i*53 + (board.middle[i] == INVALID_CARD ? 52 : board.middle[i])] = 1.0f;
        current_offset += 5 * 53;
        for(int i=0; i<5; ++i) features[current_offset + i*53 + (board.bottom[i] == INVALID_CARD ? 52 : board.bottom[i])] = 1.0f;
        current_offset += 5 * 53;
    };
    process_board(my_board, offset);
    process_board(opp_board, offset);
    const auto& my_discards = state.get_my_discards(player_view);
    for (Card c : my_discards) {
        if (c != INVALID_CARD) features[offset + c] = 1.0f;
    }
    offset += 52;
    features[offset++] = static_cast<float>(state.get_opponent_discard_count(player_view));
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

    std::vector<float> logits;
    {
        py::gil_scoped_acquire acquire;
        std::mutex m;
        std::condition_variable cv;
        bool ready = false;
        
        auto responder = [&](std::vector<float> result) {
            std::lock_guard<std::mutex> lk(m);
            logits = std::move(result);
            ready = true;
            cv.notify_one();
        };

        std::cout << "[C++ TID: " << std::this_thread::get_id() << " | TR_ID: " << traversal_id << "] Calling Policy CB. Actions: " << num_actions << std::endl << std::flush;
        policy_callback_(traversal_id, infoset_vec, canonical_action_vectors, responder);

        py::gil_scoped_release release;
        std::cout << "[C++ TID: " << std::this_thread::get_id() << " | TR_ID: " << traversal_id << "] Waiting for Policy response..." << std::endl << std::flush;
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&]{ return ready; });
        std::cout << "[C++ TID: " << std::this_thread::get_id() << " | TR_ID: " << traversal_id << "] Policy response received." << std::endl << std::flush;
    }

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
    
    float value_baseline = 0.0f;
    {
        py::gil_scoped_acquire acquire;
        std::mutex m;
        std::condition_variable cv;
        bool ready = false;

        auto responder = [&](float result) {
            std::lock_guard<std::mutex> lk(m);
            value_baseline = result;
            ready = true;
            cv.notify_one();
        };

        std::cout << "[C++ TID: " << std::this_thread::get_id() << " | TR_ID: " << traversal_id << "] Calling Value CB." << std::endl << std::flush;
        value_callback_(traversal_id, infoset_vec, responder);

        py::gil_scoped_release release;
        std::cout << "[C++ TID: " << std::this_thread::get_id() << " | TR_ID: " << traversal_id << "] Waiting for Value response..." << std::endl << std::flush;
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&]{ return ready; });
        std::cout << "[C++ TID: " << std::this_thread::get_id() << " | TR_ID: " << traversal_id << "] Value response received." << std::endl << std::flush;
    }

    value_buffer_->push(infoset_vec, dummy_action_vec_, node_payoffs.at(current_player));

    for (int i = 0; i < num_actions; ++i) {
        float advantage = action_payoffs[i].at(current_player) - value_baseline;
        policy_buffer_->push(infoset_vec, canonical_action_vectors[i], advantage);
    }

    return node_payoffs;
}

} // namespace ofc
