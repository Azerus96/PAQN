import numpy as np

# --- Константы для фичей ---
RANKS = '23456789TJQKA'
SUITS = 'shdc' # важен порядок для канонизации
RANK_TO_INT = {rank: i for i, rank in enumerate(RANKS)}
SUIT_TO_INT = {suit: i for i, suit in enumerate(SUITS)}

NUM_RANKS = 13
NUM_SUITS = 4

# --- Каналы для представления состояния ---
P_TOP_CHAN = 0
P_MID_CHAN = 1
P_BOT_CHAN = 2
P_HAND_CHAN = 3
O_TOP_CHAN = 4
O_MID_CHAN = 5
O_BOT_CHAN = 6
P_TOP_DEAD_CHAN = 7
P_MID_DEAD_CHAN = 8
P_BOT_DEAD_CHAN = 9
O_TOP_DEAD_CHAN = 10
O_MID_DEAD_CHAN = 11
O_BOT_DEAD_CHAN = 12
STREET_CHAN = 13
P_FANTASY_CHAN = 14
O_FANTASY_CHAN = 15
TURN_CHAN = 16

NUM_FEATURE_CHANNELS = 17

def card_str_to_indices(card_str: str) -> tuple[int, int]:
    """'As' -> (0, 12)"""
    if len(card_str) != 2:
        return -1, -1
    rank = RANK_TO_INT.get(card_str[0])
    suit = SUIT_TO_INT.get(card_str[1])
    if rank is None or suit is None:
        return -1, -1
    return suit, rank

def featurize_state_optimal(game_state: dict) -> np.ndarray:
    """
    Преобразует словарь состояния игры в многоканальный тензор (C, H, W).
    C = NUM_FEATURE_CHANNELS, H = NUM_SUITS, W = NUM_RANKS.
    """
    features = np.zeros((NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS), dtype=np.float32)

    # Каналы карт игрока
    for card in game_state['player_board']['top']:
        s_idx, r_idx = card_str_to_indices(card)
        if s_idx != -1: features[P_TOP_CHAN, s_idx, r_idx] = 1.0
    for card in game_state['player_board']['middle']:
        s_idx, r_idx = card_str_to_indices(card)
        if s_idx != -1: features[P_MID_CHAN, s_idx, r_idx] = 1.0
    for card in game_state['player_board']['bottom']:
        s_idx, r_idx = card_str_to_indices(card)
        if s_idx != -1: features[P_BOT_CHAN, s_idx, r_idx] = 1.0
    
    for card in game_state.get('hand', []):
        s_idx, r_idx = card_str_to_indices(card)
        if s_idx != -1: features[P_HAND_CHAN, s_idx, r_idx] = 1.0

    # Каналы карт оппонента
    for card in game_state['opponent_board']['top']:
        s_idx, r_idx = card_str_to_indices(card)
        if s_idx != -1: features[O_TOP_CHAN, s_idx, r_idx] = 1.0
    for card in game_state['opponent_board']['middle']:
        s_idx, r_idx = card_str_to_indices(card)
        if s_idx != -1: features[O_MID_CHAN, s_idx, r_idx] = 1.0
    for card in game_state['opponent_board']['bottom']:
        s_idx, r_idx = card_str_to_indices(card)
        if s_idx != -1: features[O_BOT_CHAN, s_idx, r_idx] = 1.0

    # Каналы "мертвых" рук (пока не реализовано в C++, но задел есть)
    if game_state.get('player_dead_hands', {}).get('top', False): features[P_TOP_DEAD_CHAN, :, :] = 1.0
    
    # Скалярные каналы
    features[STREET_CHAN, :, :] = (game_state.get('street', 1) - 1) / 4.0
    if game_state.get('is_player_fantasyland', False): features[P_FANTASY_CHAN, :, :] = 1.0
    if game_state.get('is_opponent_fantasyland', False): features[O_FANTASY_CHAN, :, :] = 1.0
    if game_state.get('is_player_turn', False): features[TURN_CHAN, :, :] = 1.0
    
    return features
