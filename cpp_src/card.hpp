#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <tuple> // <-- Добавить

namespace ofc {

    using Card = uint8_t;
    using CardSet = std::vector<Card>;

    constexpr Card INVALID_CARD = 255;

    // Действие: расстановка карт и карта сброса
    using Placement = std::pair<Card, std::pair<std::string, int>>;
    using Action = std::pair<std::vector<Placement>, Card>;

    // --- ДОБАВЛЕНИЕ: Операторы сравнения для Action ---
    inline bool operator<(const Placement& a, const Placement& b) {
        return std::tie(a.first, a.second.first, a.second.second) < std::tie(b.first, b.second.first, b.second.second);
    }

    inline bool operator==(const Placement& a, const Placement& b) {
        return std::tie(a.first, a.second.first, a.second.second) == std::tie(b.first, b.second.first, b.second.second);
    }

    inline bool operator<(const Action& a, const Action& b) {
        return std::tie(a.first, a.second) < std::tie(b.first, b.second);
    }

    inline bool operator==(const Action& a, const Action& b) {
        return std::tie(a.first, a.second) == std::tie(b.first, b.second);
    }
    // --- КОНЕЦ ДОБАВЛЕНИЯ ---

    inline int get_rank(Card c) { return c / 4; }
    inline int get_suit(Card c) { return c % 4; }

    inline std::string card_to_string(Card c) {
        const std::string RANKS = "23456789TJQKA";
        const std::string SUITS = "shdc";
        if (c == INVALID_CARD) return "??";
        return std::string(1, RANKS[get_rank(c)]) + std::string(1, SUITS[get_suit(c)]);
    }
}
