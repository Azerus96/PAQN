#include "game_state.hpp"
#include "constants.hpp" 
#include <iostream>
#include <map>
#include <chrono>
#include <functional>
#include <thread>
#include <set>

namespace ofc {

    GameState::GameState(int num_players, int dealer_pos)
        : num_players_(num_players) {
        deck_.reserve(52);
        dealt_cards_.reserve(5);
        my_discards_.resize(num_players);
        for(auto& v : my_discards_) v.reserve(4);
        opponent_discard_counts_.resize(num_players);
        boards_.resize(num_players);
        reset(dealer_pos);
    }

    void GameState::reset(int dealer_pos) {
        street_ = 1;
        for (auto& board : boards_) {
            board.top.fill(INVALID_CARD);
            board.middle.fill(INVALID_CARD);
            board.bottom.fill(INVALID_CARD);
        }
        
        unsigned seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                        static_cast<unsigned>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::mt19937 temp_rng(seed);

        deck_.resize(52);
        std::iota(deck_.begin(), deck_.end(), 0);
        std::shuffle(deck_.begin(), deck_.end(), temp_rng);
        
        if (dealer_pos == -1) {
            std::uniform_int_distribution<int> dist(0, num_players_ - 1);
            dealer_pos_ = dist(temp_rng);
        } else {
            this->dealer_pos_ = dealer_pos;
        }
        current_player_ = (dealer_pos_ + 1) % num_players_;
        for (auto& discards : my_discards_) {
            discards.clear();
        }
        std::fill(opponent_discard_counts_.begin(), opponent_discard_counts_.end(), 0);
        deal_cards();
    }

    std::pair<float, float> GameState::get_payoffs(const HandEvaluator& evaluator) const {
        const int SCOOP_BONUS = 3;
        const Board& p1_board = boards_[0];
        const Board& p2_board = boards_[1];
        bool p1_foul = p1_board.is_foul(evaluator);
        bool p2_foul = p2_board.is_foul(evaluator);
        int p1_royalty = p1_foul ? 0 : p1_board.get_total_royalty(evaluator);
        int p2_royalty = p2_foul ? 0 : p2_board.get_total_royalty(evaluator);

        if (p1_foul && p2_foul) return {0.0f, 0.0f};
        if (p1_foul) return {-(float)(6 + p2_royalty), (float)(6 + p2_royalty)};
        if (p2_foul) return {(float)(6 + p1_royalty), -(float)(6 + p1_royalty)};

        int line_score = 0;
        
        auto compare_lines = [&](const CardSet& p1_cards, const CardSet& p2_cards) {
            HandRank p1_rank = evaluator.evaluate(p1_cards);
            HandRank p2_rank = evaluator.evaluate(p2_cards);
            if (p1_rank < p2_rank) { return 1; } 
            else if (p2_rank < p1_rank) { return -1; }
            return 0;
        };

        int top_res = compare_lines(p1_board.get_row_cards("top"), p2_board.get_row_cards("top"));
        int mid_res = compare_lines(p1_board.get_row_cards("middle"), p2_board.get_row_cards("middle"));
        int bot_res = compare_lines(p1_board.get_row_cards("bottom"), p2_board.get_row_cards("bottom"));
        
        line_score = top_res + mid_res + bot_res;

        if (top_res > 0 && mid_res > 0 && bot_res > 0) {
            line_score += SCOOP_BONUS;
        } else if (top_res < 0 && mid_res < 0 && bot_res < 0) {
            line_score -= SCOOP_BONUS;
        }
        
        float p1_total = (float)(line_score + p1_royalty - p2_royalty);
        return {p1_total, -p1_total};
    }

    void GameState::generate_all_placements_recursive(
        const CardSet& cards_to_place,
        const std::vector<std::pair<std::string, int>>& available_slots,
        std::vector<int>& current_indices,
        int start_idx,
        int k,
        Card discarded,
        std::vector<Action>& out_actions
    ) const {
        if (current_indices.size() == (size_t)k) {
            std::vector<Placement> placement;
            placement.reserve(k);
            for (int i = 0; i < k; ++i) {
                placement.push_back({cards_to_place[i], available_slots[current_indices[i]]});
            }
            std::sort(placement.begin(), placement.end(), 
                [](const Placement& a, const Placement& b){
                    if (a.second.first != b.second.first) return a.second.first < b.second.first;
                    return a.second.second < b.second.second;
            });
            out_actions.push_back({placement, discarded});
            return;
        }

        if (start_idx >= available_slots.size()) return;

        for (size_t i = start_idx; i < available_slots.size(); ++i) {
            current_indices.push_back(i);
            generate_all_placements_recursive(cards_to_place, available_slots, current_indices, i + 1, k, discarded, out_actions);
            current_indices.pop_back();
        }
    }

    void GameState::get_legal_actions(std::vector<Action>& out_actions) const {
        out_actions.clear();
        if (is_terminal()) return;

        const Board& board = boards_[current_player_];
        std::vector<std::pair<std::string, int>> available_slots;
        available_slots.reserve(13);
        for(int i=0; i<3; ++i) if(board.top[i] == INVALID_CARD) available_slots.push_back({"top", i});
        for(int i=0; i<5; ++i) if(board.middle[i] == INVALID_CARD) available_slots.push_back({"middle", i});
        for(int i=0; i<5; ++i) if(board.bottom[i] == INVALID_CARD) available_slots.push_back({"bottom", i});

        if (street_ == 1) {
            std::vector<int> indices;
            generate_all_placements_recursive(dealt_cards_, available_slots, indices, 0, 5, INVALID_CARD, out_actions);
        } else {
            for (size_t i = 0; i < dealt_cards_.size(); ++i) {
                CardSet cards_to_place;
                Card current_discarded = dealt_cards_[i];
                for (size_t j = 0; j < dealt_cards_.size(); ++j) {
                    if (i != j) cards_to_place.push_back(dealt_cards_[j]);
                }
                std::vector<int> indices;
                generate_all_placements_recursive(cards_to_place, available_slots, indices, 0, 2, current_discarded, out_actions);
            }
        }
    }

    void GameState::apply_action(const Action& action, int player_view, UndoInfo& undo_info) {
        undo_info.action = action;
        undo_info.prev_street = street_;
        undo_info.prev_current_player = current_player_;
        undo_info.dealt_cards_before_action = dealt_cards_;

        const auto& placements = action.first;
        const Card& discarded_card = action.second;

        for (const auto& p : placements) {
            const Card& card = p.first;
            const std::string& row = p.second.first;
            int idx = p.second.second;
            if (row == "top") boards_[current_player_].top[idx] = card;
            else if (row == "middle") boards_[current_player_].middle[idx] = card;
            else if (row == "bottom") boards_[current_player_].bottom[idx] = card;
        }

        if (discarded_card != INVALID_CARD) {
            if (current_player_ == player_view) {
                my_discards_[current_player_].push_back(discarded_card);
            } else {
                opponent_discard_counts_[player_view]++;
            }
        }

        if (current_player_ == dealer_pos_) street_++;
        current_player_ = (current_player_ + 1) % num_players_;
        
        if (!is_terminal()) {
            deal_cards();
        } else {
            dealt_cards_.clear();
        }
    }

    void GameState::undo_action(const UndoInfo& undo_info, int player_view) {
        street_ = undo_info.prev_street;
        current_player_ = undo_info.prev_current_player;

        deck_.insert(deck_.end(), dealt_cards_.begin(), dealt_cards_.end());
        dealt_cards_ = undo_info.dealt_cards_before_action;

        const auto& placements = undo_info.action.first;
        const Card& discarded_card = undo_info.action.second;

        for (const auto& p : placements) {
            const std::string& row = p.second.first;
            int idx = p.second.second;
            if (row == "top") boards_[current_player_].top[idx] = INVALID_CARD;
            else if (row == "middle") boards_[current_player_].middle[idx] = INVALID_CARD;
            else if (row == "bottom") boards_[current_player_].bottom[idx] = INVALID_CARD;
        }

        if (discarded_card != INVALID_CARD) {
            if (current_player_ == player_view) {
                my_discards_[current_player_].pop_back();
            } else {
                opponent_discard_counts_[player_view]--;
            }
        }
    }

    void GameState::deal_cards() {
        int num_to_deal = (street_ == 1) ? 5 : 3;
        if (deck_.size() < (size_t)num_to_deal) {
            street_ = 6; 
            dealt_cards_.clear();
            return;
        }
        dealt_cards_.assign(deck_.end() - num_to_deal, deck_.end());
        deck_.resize(deck_.size() - num_to_deal);
    }

    GameState GameState::get_canonical(std::map<int, int>& suit_map) const {
        suit_map.clear();
        GameState canonical_state = *this;
        int transform[SUIT_COUNT] = {-1, -1, -1, -1};
        int canonical_suit_count = 0;

        auto process_card_for_mapping = [&](Card card) {
            if (card == INVALID_CARD) return;
            int original_suit = get_suit(card);
            if (transform[original_suit] == -1) {
                transform[original_suit] = canonical_suit_count++;
            }
        };

        for (const auto& card : dealt_cards_) process_card_for_mapping(card);
        for (int p_idx = 0; p_idx < num_players_; ++p_idx) {
            for (const auto& card : boards_[p_idx].get_all_cards()) {
                process_card_for_mapping(card);
            }
            for (const auto& card : my_discards_[p_idx]) {
                process_card_for_mapping(card);
            }
        }

        for (int i = 0; i < SUIT_COUNT; ++i) {
            if (transform[i] == -1) {
                transform[i] = canonical_suit_count++;
            }
        }

        for (int i = 0; i < SUIT_COUNT; ++i) {
            suit_map[i] = transform[i];
        }

        auto remap_card = [&](Card& card) {
            if (card == INVALID_CARD) return;
            card = get_rank(card) * 4 + transform[get_suit(card)];
        };

        for (auto& card : canonical_state.dealt_cards_) remap_card(card);
        for (auto& board : canonical_state.boards_) {
            for (auto& card : board.top) remap_card(card);
            for (auto& card : board.middle) remap_card(card);
            for (auto& card : board.bottom) remap_card(card);
        }
        for (auto& discard_set : canonical_state.my_discards_) {
            for (auto& card : discard_set) remap_card(card);
        }

        std::sort(canonical_state.dealt_cards_.begin(), canonical_state.dealt_cards_.end());
        
        return canonical_state;
    }
}
