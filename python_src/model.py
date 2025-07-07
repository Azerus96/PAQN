import torch
import torch.nn as nn
import torch.nn.functional as F

class PAQN_Network(nn.Module):
    """
    Объединенная сеть для Policy и Value.
    Имеет общий "ствол" (body) и две "головы" (heads) - одну для политики, другую для ценности.
    """
    def __init__(self, infoset_size, action_vec_size, hidden_size=512):
        super(PAQN_Network, self).__init__()
        
        # Общий ствол для обработки инфосета
        self.body = nn.Sequential(
            nn.Linear(infoset_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Голова для предсказания ценности (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Голова для предсказания политики (Policy Head)
        # Принимает на вход выход ствола + вектор действия
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size + action_vec_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, infoset, action_vec=None):
        # Если action_vec не предоставлен, мы считаем только value
        if action_vec is None:
            body_out = self.body(infoset)
            value = self.value_head(body_out)
            return value

        # Если action_vec есть, считаем и value, и policy
        body_out = self.body(infoset)
        value = self.value_head(body_out)
        
        # Для policy head конкатенируем выход ствола с вектором действия
        policy_input = torch.cat([body_out, action_vec], dim=1)
        policy_logit = self.policy_head(policy_input)
        
        return policy_logit, value

# <<< ИЗМЕНЕНИЕ: Старые классы можно удалить или закомментировать,
# так как они больше не используются в train.py.
# class PolicyNetwork(...): ...
# class ValueNetwork(...): ...
