#include "DeepMCCFR.hpp"
#include "constants.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <map>
#include <random> // <-- ВАЖНО: Добавлен необходимый заголовок

namespace ofc {

// Вспомогательная функция для преобразования ofc::Action в вектор
// (Эта функция остается без изменений)
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
        vec[discarded_card * 4 + 3] = 1.0f; // Слот 3 для сброса
    }
    return vec;
}

// --- НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ---
// Реализует добавление шума Дирихле к вектору стратегии для обеспечения исследования.
void add_dirichlet_noise(std::vector<float>& strategy, float alpha, std::mt19937& rng) {
    if (strategy.empty()) {
        return;
    }
    
    // 1. Генерируем шум из гамма-распределения
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(strategy.size());
    float noise_sum = 0.0f;

    for (size_t i = 0; i < strategy.size(); ++i) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }

    // 2. Нормализуем шум и смешиваем с исходной стратегией
    if (noise_sum > 1e-6) { // Защита от деления на ноль
        const float exploration_fraction = 0.25f; // Доля шума в итоговой стратегии
        for (size_t i = 0; i < strategy.size(); ++i) {
            strategy[i] = (1.0f - exploration_fraction) * strategy[i] + exploration_fraction * (noise[i] / noise_sum);
        }
    }
}


DeepMCCFR::DeepMCCFR(size_t action_limit, SharedReplayBuffer* buffer, InferenceQueue* queue) 
    : action_limit_(action_limit), replay_buffer_(buffer), inference_queue_(queue), rng_(std::random_device{}()) {
}

void DeepMCCFR::run_traversal() {
    GameState state; 
    // Вызываем обход для первого игрока, указывая, что это корневой узел (is_root = true)
    traverse(state, 0, true);
    state.reset(); 
    // Вызываем обход для второго игрока, также указывая, что это корневой узел
    traverse(state, 1, true);
}

std::vector<float> DeepMCCFR::featurize(const GameState& state, int player_view) {
    // (Эта функция остается без изменений)
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
            Card c = board.top[i];
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 3 * 53;
        for(int i=0; i<5; ++i) {
            Card c = board.middle[i];
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 5 * 53;
        for(int i=0; i<5; ++i) {
            Card c = board.bottom[i];
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
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

// --- ИЗМЕНЕНИЕ: Обновлена сигнатура и логика функции traverse ---
std::map<int, float> DeepMCCFR::traverse(GameState& state, int traversing_player, bool is_root) {
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
        // Все рекурсивные вызовы теперь передают is_root = false
        auto result = traverse(state, traversing_player, false);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    if (current_player != traversing_player) {
        int action_idx = std::uniform_int_distribution<int>(0, num_actions - 1)(rng_);
        state.apply_action(legal_actions[action_idx], traversing_player, undo_info);
        // Все рекурсивные вызовы теперь передают is_root = false
        auto result = traverse(state, traversing_player, false);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    // --- Логика инференса (без изменений) ---
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
        for (auto& placement : canonical_action.first) {
            remap_card(placement.first);
        }
        remap_card(canonical_action.second);
        canonical_action_vectors.push_back(action_to_vector(canonical_action));
    }
    std::vector<float> regrets;
    {
        std::promise<std::vector<float>> promise;
        std::future<std::vector<float>> future = promise.get_future();
        InferenceRequest request;
        request.infoset = infoset_vec;
        request.action_vectors = std::move(canonical_action_vectors); 
        request.promise = std::move(promise);
        inference_queue_->push(std::move(request));
        regrets = future.get();
    }

    // --- ПОЛНОСТЬЮ НОВАЯ ЛОГИКА ФОРМИРОВАНИЯ СТРАТЕГИИ ---
    // 5. Вычисляем стратегию на основе предсказанных сожалений (Regret Matching)
    std::vector<float> strategy(num_actions);
    float total_positive_regret = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        strategy[i] = (regrets[i] > 0) ? regrets[i] : 0.0f;
        total_positive_regret += strategy[i];
    }

    if (total_positive_regret > 1e-6) { // Используем порог для численной стабильности
        for (int i = 0; i < num_actions; ++i) {
            strategy[i] /= total_positive_regret;
        }
    } else {
        // Если нет положительных сожалений (например, в начале обучения), играем случайно
        std::fill(strategy.begin(), strategy.end(), 1.0f / num_actions);
    }

    // Добавляем шум Дирихле для исследования, но ТОЛЬКО в корневом узле обхода
    if (is_root) {
        const float DIRICHLET_ALPHA = 0.3f; // Гиперпараметр, контролирующий "силу" шума
        add_dirichlet_noise(strategy, DIRICHLET_ALPHA, rng_);
    }
    // --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    // 6. Обходим дочерние узлы и вычисляем истинные сожаления
    std::vector<std::map<int, float>> action_utils(num_actions);
    std::map<int, float> node_util = {{0, 0.0f}, {1, 0.0f}};

    for (int i = 0; i < num_actions; ++i) {
        state.apply_action(legal_actions[i], traversing_player, undo_info);
        // Все рекурсивные вызовы теперь передают is_root = false
        action_utils[i] = traverse(state, traversing_player, false);
        state.undo_action(undo_info, traversing_player);

        for(auto const& [player_idx, util] : action_utils[i]) {
            node_util[player_idx] += strategy[i] * util;
        }
    }

    std::vector<float> true_regrets(num_actions);
    for(int i = 0; i < num_actions; ++i) {
        true_regrets[i] = action_utils[i][current_player] - node_util[current_player];
    }
    
    // 7. Сохраняем данные в буфер (этот блок без изменений)
    for (int i = 0; i < num_actions; ++i) {
        Action canonical_action = legal_actions[i];
        std::map<int, int> current_suit_map;
        state.get_canonical(current_suit_map);
        auto remap_card_for_save = [&](Card& card) {
            if (card == INVALID_CARD) return;
            card = get_rank(card) * 4 + current_suit_map.at(get_suit(card));
        };
        for (auto& placement : canonical_action.first) {
            remap_card_for_save(placement.first);
        }
        remap_card_for_save(canonical_action.second);
        std::vector<float> action_vec = action_to_vector(canonical_action);
        replay_buffer_->push(infoset_vec, action_vec, true_regrets[i]);
    }

    return node_util;
}

}
