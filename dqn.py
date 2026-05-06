import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 96),
            nn.LayerNorm(96),
            nn.ReLU(),

            nn.Linear(96, 96),
            nn.LayerNorm(96),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self._init_weights()

    def forward(self, x):
        x = x.float()

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)