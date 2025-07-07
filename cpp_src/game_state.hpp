#pragma once
#include "board.hpp"
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <map>
#include <string>

namespace ofc {

    // Структура для отмены действия
    struct UndoInfo {
        Action action;
        int prev_street;
        int prev_current_player;
        CardSet dealt_cards_before_action;
    };

    // Класс, представляющий полное состояние игры
    class GameState {
    public:
        GameState(int num_players = 2, int dealer_pos = -1);
        
        // Сбрасывает состояние игры к началу новой раздачи
        void reset(int dealer_pos = -1);

        // Проверяет, является ли состояние терминальным (конец игры)
        inline bool is_terminal() const {
            return street_ > 5 || boards_[0].get_card_count() == 13;
        }

        // Рассчитывает итоговые очки для двух игроков
        std::pair<float, float> get_payoffs(const HandEvaluator& evaluator) const;
        
        // Генерирует список легальных действий для текущего игрока
        void get_legal_actions(size_t action_limit, std::vector<Action>& out_actions, std::mt19937& rng) const;
        
        // Применяет действие к состоянию игры
        void apply_action(const Action& action, int player_view, UndoInfo& undo_info);
        
        // Отменяет последнее примененное действие
        void undo_action(const UndoInfo& undo_info, int player_view);
        
        // Возвращает каноническое представление состояния
        // ИЗМЕНЕНО: Метод теперь принимает легальные действия для построения полной карты мастей
        GameState get_canonical(const std::vector<Action>& legal_actions, std::map<int, int>& suit_map) const;

        // Геттеры для получения информации о состоянии
        int get_street() const { return street_; }
        int get_current_player() const { return current_player_; }
        const CardSet& get_dealt_cards() const { return dealt_cards_; }
        const Board& get_player_board(int player_idx) const { return boards_[player_idx]; }
        const Board& get_opponent_board(int player_idx) const { return boards_[(player_idx + 1) % num_players_]; }
        const CardSet& get_my_discards(int player_idx) const { return my_discards_[player_idx]; }
        int get_opponent_discard_count(int player_idx) const { return opponent_discard_counts_[player_idx]; }
        int get_dealer_pos() const { return dealer_pos_; }
        
    private:
        // Внутренние вспомогательные методы
        void deal_cards();
        void generate_random_placements(const CardSet& cards, Card discarded, std::vector<Action>& actions, size_t limit, std::mt19937& rng) const;

        // Поля класса
        int num_players_;
        int street_;
        int dealer_pos_;
        int current_player_;
        std::vector<Board> boards_;
        CardSet deck_;
        CardSet dealt_cards_;
        std::vector<CardSet> my_discards_;
        std::vector<int> opponent_discard_counts_;
    };
}
