import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Сеть Политики (Policy Network).
    Предсказывает "логит" (не нормализованный скор) для пары (состояние, действие).
    Чем выше логит, тем лучше считается действие в данном состоянии.
    """
    def __init__(self, infoset_size, action_vec_size, hidden_size=512):
        super(PolicyNetwork, self).__init__()
        input_layer_size = infoset_size + action_vec_size
        
        self.net = nn.Sequential(
            nn.Linear(input_layer_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1) # Выход - один логит
        )

    def forward(self, infoset, action_vec):
        combined_input = torch.cat([infoset, action_vec], dim=1)
        return self.net(combined_input)


class ValueNetwork(nn.Module):
    """
    Сеть Ценности (Value Network).
    Предсказывает ценность (ожидаемый payoff) для данного состояния (инфосета).
    """
    def __init__(self, infoset_size, hidden_size=512):
        super(ValueNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(infoset_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1) # Выход - одна оценка ценности
        )

    def forward(self, infoset):
        return self.net(infoset)
