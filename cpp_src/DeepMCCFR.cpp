#include "DeepMCCFR.hpp"
#include "constants.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <map>

namespace ofc {

// Вспомогательная функция для преобразования ofc::Action в вектор
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


DeepMCCFR::DeepMCCFR(size_t action_limit, SharedReplayBuffer* buffer, InferenceQueue* queue) 
    : action_limit_(action_limit), replay_buffer_(buffer), inference_queue_(queue), rng_(std::random_device{}()) {
}

void DeepMCCFR::run_traversal() {
    GameState state; 
    traverse(state, 0);
    state.reset(); 
    traverse(state, 1);
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

std::map<int, float> DeepMCCFR::traverse(GameState& state, int traversing_player) {
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
        auto result = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    if (current_player != traversing_player) {
        int action_idx = std::uniform_int_distribution<int>(0, num_actions - 1)(rng_);
        state.apply_action(legal_actions[action_idx], traversing_player, undo_info);
        auto result = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    // --- НОВАЯ ЛОГИКА С ПОЛНОЦЕННЫМ ИНФЕРЕНСОМ ---
    // 1. Создаем карту для трансформации и получаем каноническое состояние
    std::map<int, int> suit_map;
    GameState canonical_state = state.get_canonical(suit_map);

    // 2. Создаем инфосет из канонического состояния
    std::vector<float> infoset_vec = featurize(canonical_state, traversing_player);
    
    // 3. Канонизируем все легальные действия и создаем их векторы
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

    // 4. Отправляем запрос в нейросеть
    std::vector<float> regrets;
    {
        std::promise<std::vector<float>> promise;
        std::future<std::vector<float>> future = promise.get_future();

        InferenceRequest request;
        request.infoset = infoset_vec;
        // Передаем созданные векторы. Используем std::move для эффективности.
        request.action_vectors = std::move(canonical_action_vectors); 
        request.promise = std::move(promise);
        
        inference_queue_->push(std::move(request));

        regrets = future.get(); // Ждем предсказанные сожаления
    }

    // 5. Вычисляем стратегию на основе предсказанных сожалений
    std::vector<float> strategy(num_actions);
    float total_positive_regret = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        strategy[i] = (regrets[i] > 0) ? regrets[i] : 0.0f;
        total_positive_regret += strategy[i];
    }

    if (total_positive_regret > 0) {
        for (int i = 0; i < num_actions; ++i) strategy[i] /= total_positive_regret;
    } else {
        std::fill(strategy.begin(), strategy.end(), 1.0f / num_actions);
    }

    // 6. Обходим дочерние узлы и вычисляем истинные сожаления
    std::vector<std::map<int, float>> action_utils(num_actions);
    std::map<int, float> node_util = {{0, 0.0f}, {1, 0.0f}};

    for (int i = 0; i < num_actions; ++i) {
        state.apply_action(legal_actions[i], traversing_player, undo_info);
        action_utils[i] = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);

        for(auto const& [player_idx, util] : action_utils[i]) {
            node_util[player_idx] += strategy[i] * util;
        }
    }

    std::vector<float> true_regrets(num_actions);
    for(int i = 0; i < num_actions; ++i) {
        true_regrets[i] = action_utils[i][current_player] - node_util[current_player];
    }
    
    // 7. Сохраняем данные в буфер (инфосет, канонизированное действие, истинное сожаление)
    // Векторы канонизированных действий мы уже отправили в очередь, и они там уничтожились.
    // Поэтому нам нужно их создать заново для сохранения в буфер.
    // Это небольшая избыточность, но она упрощает код.
    for (int i = 0; i < num_actions; ++i) {
        Action canonical_action = legal_actions[i];
        for (auto& placement : canonical_action.first) {
            remap_card(placement.first);
        }
        remap_card(canonical_action.second);
        std::vector<float> action_vec = action_to_vector(canonical_action);
        replay_buffer_->push(infoset_vec, action_vec, true_regrets[i]);
    }
    // --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    return node_util;
}

}
