import torch
import torch.nn as nn
import torch.nn.functional as F

from cpp_src.constants import NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS, ACTION_VECTOR_SIZE

class OFC_CNN_Network(nn.Module):
    def __init__(self, hidden_size=512):
        super(OFC_CNN_Network, self).__init__()
        
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

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        policy_input_size = hidden_size + ACTION_VECTOR_SIZE + 5 
        
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state_image, action_vec=None, street_vector=None):
        conv_out = self.body_conv(state_image)
        flat_out = conv_out.view(conv_out.size(0), -1)
        body_out = self.body_fc(flat_out)
        
        if action_vec is None:
            value = self.value_head(body_out)
            return value
        
        if street_vector is None:
            raise ValueError("street_vector must be provided for policy evaluation")
            
        policy_input = torch.cat([body_out, action_vec, street_vector], dim=1)
        policy_logit = self.policy_head(policy_input)
        
        return policy_logit, None
