// D2CFR-main/cpp_src/hand_evaluator.hpp (ВЕРСИЯ 5.0)

#pragma once
#include "card.hpp"
#include <omp/HandEvaluator.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>
#include <array>

namespace ofc {

    struct HandRank {
        int rank_value;
        int hand_class; // 6 = Trips, 8 = Pair, 9 = High Card
        std::string type_str;

        bool operator<(const HandRank& other) const {
            return rank_value < other.rank_value;
        }
    };

    class HandEvaluator {
    public:
        HandEvaluator() {
            init_3_card_lookup();
        }

        inline HandRank evaluate(const CardSet& cards) const {
            if (cards.size() == 5) {
                omp::Hand h = omp::Hand::empty();
                for (Card c : cards) h += omp::Hand(c);
                int rank_value = evaluator_5_card_.evaluate(h);
                int hand_class_omp = rank_value >> 12;
                
                int hand_class = (hand_class_omp > 0) ? (10 - hand_class_omp) : 9;
                
                static const std::map<int, std::string> class_to_string_map_5 = {
                    {1, "Straight Flush"}, {2, "Four of a Kind"}, {3, "Full House"},
                    {4, "Flush"}, {5, "Straight"}, {6, "Three of a Kind"},
                    {7, "Two Pair"}, {8, "Pair"}, {9, "High Card"}
                };
                return {rank_value, hand_class, class_to_string_map_5.at(hand_class)};
            }
            if (cards.size() == 3) {
                std::array<int, 3> ranks = {get_rank(cards[0]), get_rank(cards[1]), get_rank(cards[2])};
                std::sort(ranks.rbegin(), ranks.rend());
                int key = ranks[0] * 169 + ranks[1] * 13 + ranks[2];
                return evaluator_3_card_lookup_[key];
            }
            return {9999, 9, "Invalid"};
        }

        inline int get_royalty(const CardSet& cards, const std::string& row_name) const {
            static const std::unordered_map<std::string, int> ROYALTY_BOTTOM = {{"Straight", 2}, {"Flush", 4}, {"Full House", 6}, {"Four of a Kind", 10}, {"Straight Flush", 15}, {"Royal Flush", 25}};
            static const std::unordered_map<std::string, int> ROYALTY_MIDDLE = {{"Three of a Kind", 2}, {"Straight", 4}, {"Flush", 8}, {"Full House", 12}, {"Four of a Kind", 20}, {"Straight Flush", 30}, {"Royal Flush", 50}};
            static const std::array<int, 13> ROYALTY_TOP_PAIRS = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 10, 11, 12};
            static const std::array<int, 13> ROYALTY_TOP_TRIPS = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};

            if (cards.empty()) return 0;
            HandRank hr = evaluate(cards);

            if (row_name == "top") {
                if (hr.hand_class == 6) { // Trips
                    int rank = get_rank(cards[0]);
                    return ROYALTY_TOP_TRIPS[rank];
                } else if (hr.hand_class == 8) { // Pair
                    std::array<int, 3> ranks = {get_rank(cards[0]), get_rank(cards[1]), get_rank(cards[2])};
                    int pair_rank = (ranks[0] == ranks[1] || ranks[0] == ranks[2]) ? ranks[0] : ranks[1];
                    if (pair_rank >= 4) return ROYALTY_TOP_PAIRS[pair_rank];
                }
            } else if (row_name == "middle") {
                auto it = ROYALTY_MIDDLE.find(hr.type_str);
                return (it != ROYALTY_MIDDLE.end()) ? it->second : 0;
            } else if (row_name == "bottom") {
                auto it = ROYALTY_BOTTOM.find(hr.type_str);
                return (it != ROYALTY_BOTTOM.end()) ? it->second : 0;
            }
            return 0;
        }

    private:
        omp::HandEvaluator evaluator_5_card_;
        std::array<HandRank, 2197> evaluator_3_card_lookup_;

        void init_3_card_lookup() {
            for (int r = 0; r <= 12; ++r) {
                int key = r * 169 + r * 13 + r;
                evaluator_3_card_lookup_[key] = {1000 + (12-r), 6, "Trips"};
            }
            int rank_val = 2000;
            for (int p = 12; p >= 0; --p) {
                for (int k = 12; k >= 0; --k) {
                    if (p == k) continue;
                    std::array<int, 3> ranks = {p, p, k};
                    std::sort(ranks.rbegin(), ranks.rend());
                    int key = ranks[0] * 169 + ranks[1] * 13 + ranks[2];
                    evaluator_3_card_lookup_[key] = {rank_val++, 8, "Pair"};
                }
            }
            rank_val = 3000;
            for (int r1 = 12; r1 >= 2; --r1) {
                for (int r2 = r1 - 1; r2 >= 1; --r2) {
                    for (int r3 = r2 - 1; r3 >= 0; --r3) {
                        int key = r1 * 169 + r2 * 13 + r3;
                        evaluator_3_card_lookup_[key] = {rank_val++, 9, "High Card"};
                    }
                }
            }
        }
    };
}
