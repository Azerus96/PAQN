#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>

namespace ofc {

    using Card = uint8_t;
    using CardSet = std::vector<Card>;

    constexpr Card INVALID_CARD = 255;

    // Действие: расстановка карт и карта сброса
    using Placement = std::pair<Card, std::pair<std::string, int>>;
    using Action = std::pair<std::vector<Placement>, Card>;

    inline int get_rank(Card c) { return c / 4; }
    inline int get_suit(Card c) { return c % 4; }

    inline std::string card_to_string(Card c) {
        const std::string RANKS = "23456789TJQKA";
        const std::string SUITS = "shdc";
        if (c == INVALID_CARD) return "??";
        return std::string(1, RANKS[get_rank(c)]) + std::string(1, SUITS[get_suit(c)]);
    }
}
