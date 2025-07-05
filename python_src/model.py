import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRegretNetwork(nn.Module):
    """
    Простая нейросетевая архитектура для прямого предсказания сожалений (regrets).
    Она принимает на вход конкатенированный вектор инфосета и действия и выдает одно число.
    Эта архитектура заменяет более сложную и потенциально нестабильную Dueling-архитектуру.
    """
    def __init__(self, infoset_size, action_vec_size, hidden_size=512):
        super(SimpleRegretNetwork, self).__init__()
        
        # Входной слой принимает конкатенацию инфосета и вектора действия
        input_layer_size = infoset_size + action_vec_size
        
        self.net = nn.Sequential(
            nn.Linear(input_layer_size, hidden_size),
            nn.ReLU(),
            # Можно добавить больше слоев для увеличения глубины
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # Выходной слой выдает одно скалярное значение - предсказанное сожаление
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, infoset, action_vec):
        """
        Принимает на вход батч инфосетов и батч векторов действий.
        """
        # 1. Конкатенируем инфосет и вектор действия по второму измерению (dim=1)
        combined_input = torch.cat([infoset, action_vec], dim=1)
        
        # 2. Прогоняем через сеть для получения предсказания
        return self.net(combined_input)
