#include "game_state.hpp"
#include "constants.hpp" 
#include <iostream>
#include <map>
#include <chrono>
#include <functional> // Для std::hash
#include <thread>     // Для std::thread::id

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
        if (p1_foul) return {-(float)(SCOOP_BONUS + p2_royalty), (float)(SCOOP_BONUS + p2_royalty)};
        if (p2_foul) return {(float)(SCOOP_BONUS + p1_royalty), -(float)(SCOOP_BONUS + p1_royalty)};

        int line_score = 0;
        
        auto compare_lines = [&](const CardSet& p1_cards, const CardSet& p2_cards) {
            HandRank p1_rank = evaluator.evaluate(p1_cards);
            HandRank p2_rank = evaluator.evaluate(p2_cards);
            if (p1_rank < p2_rank) {
                line_score++;
            } else if (p2_rank < p1_rank) {
                line_score--;
            }
        };

        compare_lines(p1_board.get_row_cards("top"), p2_board.get_row_cards("top"));
        compare_lines(p1_board.get_row_cards("middle"), p2_board.get_row_cards("middle"));
        compare_lines(p1_board.get_row_cards("bottom"), p2_board.get_row_cards("bottom"));

        if (abs(line_score) == 3) {
            line_score = (line_score > 0) ? SCOOP_BONUS : -SCOOP_BONUS;
        }
        
        float p1_total = (float)(line_score + p1_royalty - p2_royalty);
        return {p1_total, -p1_total};
    }

    void GameState::get_legal_actions(size_t action_limit, std::vector<Action>& out_actions, std::mt19937& rng) const {
        out_actions.clear();
        if (is_terminal()) return;

        CardSet cards_to_place;
        cards_to_place.reserve(5);
        if (street_ == 1) {
            cards_to_place = dealt_cards_;
            generate_random_placements(cards_to_place, INVALID_CARD, out_actions, action_limit, rng);
        } else {
            size_t limit_per_discard = (action_limit > 0 && dealt_cards_.size() > 1) ? (action_limit / dealt_cards_.size() + 1) : action_limit;
            for (size_t i = 0; i < dealt_cards_.size(); ++i) {
                cards_to_place.clear();
                Card current_discarded = dealt_cards_[i];
                for (size_t j = 0; j < dealt_cards_.size(); ++j) {
                    if (i != j) cards_to_place.push_back(dealt_cards_[j]);
                }
                generate_random_placements(cards_to_place, current_discarded, out_actions, limit_per_discard, rng);
            }
        }
        
        if (action_limit > 0 && out_actions.size() > action_limit) {
            std::shuffle(out_actions.begin(), out_actions.end(), rng);
            out_actions.resize(action_limit);
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

    void GameState::generate_random_placements(const CardSet& cards, Card discarded, std::vector<Action>& actions, size_t limit, std::mt19937& rng) const {
        const Board& board = boards_[current_player_];
        std::vector<std::pair<std::string, int>> available_slots;
        available_slots.reserve(13);
        for(int i=0; i<3; ++i) if(board.top[i] == INVALID_CARD) available_slots.push_back({"top", i});
        for(int i=0; i<5; ++i) if(board.middle[i] == INVALID_CARD) available_slots.push_back({"middle", i});
        for(int i=0; i<5; ++i) if(board.bottom[i] == INVALID_CARD) available_slots.push_back({"bottom", i});

        size_t k = cards.size();
        if (available_slots.size() < k) return;

        CardSet temp_cards = cards;
        std::vector<std::pair<std::string, int>> temp_slots = available_slots;
        
        std::vector<Placement> current_placement;
        current_placement.reserve(k);

        for (size_t i = 0; i < limit; ++i) {
            std::shuffle(temp_cards.begin(), temp_cards.end(), rng);
            std::shuffle(temp_slots.begin(), temp_slots.end(), rng);

            current_placement.clear();
            for(size_t j = 0; j < k; ++j) {
                current_placement.push_back({temp_cards[j], temp_slots[j]});
            }
            
            std::sort(current_placement.begin(), current_placement.end(), 
                [](const Placement& a, const Placement& b){
                    if (a.second.first != b.second.first) return a.second.first < b.second.first;
                    return a.second.second < b.second.second;
            });

            actions.push_back({current_placement, discarded});
        }
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
