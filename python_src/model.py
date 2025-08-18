import torch
import torch.nn as nn
import torch.nn.functional as F

# --- КОНСТАНТЫ ---
NUM_FEATURE_CHANNELS = 16
NUM_SUITS = 4
NUM_RANKS = 13
ACTION_VECTOR_SIZE = 208

class ResBlock(nn.Module):
    """
    Остаточный блок с GroupNorm для стабильного извлечения признаков.
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.gn1 = nn.GroupNorm(groups, channels)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.gn1(x))
        out = self.conv1(out)
        out = self.act(self.gn2(out))
        out = self.conv2(out)
        out = out + identity # Ключевая остаточная связь
        return out

class OFC_CNN_Network(nn.Module):
    """
    Улучшенная архитектура сети с ResNet-подобным телом и сбалансированной policy-головой.
    """
    def __init__(self, hidden_size=512, channels=64, num_res_blocks=6, groups=8):
        super(OFC_CNN_Network, self).__init__()
        
        # 1. Сверточное "тело" для извлечения признаков из состояния игры
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_FEATURE_CHANNELS, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
        )
        self.res_trunk = nn.Sequential(*[ResBlock(channels, groups) for _ in range(num_res_blocks)])
        
        conv_output_size = channels * NUM_SUITS * NUM_RANKS
        
        # 2. Общая полносвязная часть
        self.body_fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.SiLU(inplace=True),
        )

        # 3. Голова для оценки ценности состояния (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 4. Голова для оценки действий (Policy Head)
        # Проекции для векторов действия и улицы для уменьшения размерности
        self.action_proj = nn.Sequential(
            nn.LayerNorm(ACTION_VECTOR_SIZE),
            nn.Linear(ACTION_VECTOR_SIZE, 128),
            nn.SiLU(inplace=True),
        )
        self.street_proj = nn.Sequential(
            nn.Linear(5, 8),
            nn.SiLU(inplace=True),
        )
        
        policy_input_size = hidden_size + 128 + 8
        self.policy_ln = nn.LayerNorm(policy_input_size) # LayerNorm для стабилизации входа
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state_image, action_vec=None, street_vector=None):
        # Прогон через сверточную часть
        x = self.stem(state_image)
        x = self.res_trunk(x)
        
        # Выпрямление и прогон через общую полносвязную часть
        flat_out = x.view(x.size(0), -1)
        body_out = self.body_fc(flat_out)
        
        # Если нужен только Value (оценка состояния)
        if action_vec is None:
            value = self.value_head(body_out)
            return value
        
        # Если нужен Policy (оценка действия)
        if street_vector is None:
            raise ValueError("street_vector must be provided for policy evaluation")
            
        # Проекция и конкатенация всех входов для policy-головы
        action_embedding = self.action_proj(action_vec)
        street_embedding = self.street_proj(street_vector)
        
        policy_input = torch.cat([body_out, action_embedding, street_embedding], dim=1)
        policy_input_normalized = self.policy_ln(policy_input)
        policy_logit = self.policy_head(policy_input_normalized)
        
        return policy_logit, None
