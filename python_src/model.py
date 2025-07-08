import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ИЗМЕНЕНИЕ ---: Константы для новой CNN-модели
NUM_FEATURE_CHANNELS = 17
NUM_SUITS = 4
NUM_RANKS = 13
ACTION_VECTOR_SIZE = 208

class OFC_CNN_Network(nn.Module):
    def __init__(self, hidden_size=512):
        super(OFC_CNN_Network, self).__init__()
        
        # Сверточный ствол для "зрения"
        self.body_conv = nn.Sequential(
            nn.Conv2d(in_channels=NUM_FEATURE_CHANNELS, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        conv_output_size = 64 * NUM_SUITS * NUM_RANKS
        
        self.body_fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU()
        )

        # Голова для предсказания ценности (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Голова для предсказания политики (Policy Head)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size + ACTION_VECTOR_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state_image, action_vec=None):
        conv_out = self.body_conv(state_image)
        flat_out = conv_out.view(conv_out.size(0), -1)
        body_out = self.body_fc(flat_out)
        
        if action_vec is None:
            value = self.value_head(body_out)
            return value

        value = self.value_head(body_out)
        
        policy_input = torch.cat([body_out, action_vec], dim=1)
        policy_logit = self.policy_head(policy_input)
        
        return policy_logit, value
