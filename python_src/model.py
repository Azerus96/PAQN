import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricDuelingNetwork(nn.Module):
    """
    Гибридная архитектура, сочетающая Dueling-подход с параметрическим входом для действий (PAQN).
    """
    def __init__(self, infoset_size, action_vec_size, hidden_size=512):
        super(ParametricDuelingNetwork, self).__init__()

        # --- Поток для оценки состояния (Value Stream) ---
        # Этот поток смотрит ТОЛЬКО на инфосет, чтобы оценить ценность позиции в целом.
        self.state_feature_layer = nn.Sequential(
            nn.Linear(infoset_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(hidden_size // 2, 1)

        # --- Поток для оценки преимущества (Advantage Stream) ---
        # Этот поток смотрит на инфосет И на конкретное действие, чтобы оценить,
        # насколько это действие лучше "среднего" в данной ситуации.
        self.action_feature_layer = nn.Sequential(
            nn.Linear(infoset_size + action_vec_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.advantage_stream = nn.Linear(hidden_size // 2, 1)

    def forward(self, infoset, action_vec):
        """
        Принимает на вход батч инфосетов и батч векторов действий.
        """
        # 1. Вычисляем ценность состояния (V(s))
        state_features = self.state_feature_layer(infoset)
        value = self.value_stream(state_features)

        # 2. Конкатенируем инфосет и действие для потока преимущества
        combined_input = torch.cat([infoset, action_vec], dim=1)
        
        # 3. Вычисляем преимущество действия (A(s, a))
        action_features = self.action_feature_layer(combined_input)
        advantage = self.advantage_stream(action_features)
        
        # 4. Итоговая Q-ценность (сожаление) = V(s) + A(s, a)
        # В PAQN нет "вычитания среднего", так как мы оцениваем одно действие за раз.
        # Dueling-идея здесь в том, что мы разделяем знание о ценности состояния
        # и знание о преимуществе конкретного действия.
        q_value = value + advantage
        
        return q_value
