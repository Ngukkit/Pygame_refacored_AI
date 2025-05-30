import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQNModel(nn.Module):
    def __init__(self, state_size, action_size, target_count):
        super(DuelingDQNModel, self).__init__()
        self.input_size = state_size + target_count  # target_id one-hot 포함

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_size)
        )


    def forward(self, state, target_onehot, return_hidden=False):
        x = torch.cat([state, target_onehot], dim=1)
        x = F.silu(self.fc1(x))
        h1 = x
        x = F.silu(self.fc2(x))
        h2 = x
        x = F.silu(self.fc3(x))
        h3 = x
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        if return_hidden:
            return q, h1, h2, h3
        else:
            return q

class RewardNet(nn.Module):
    def __init__(self, state_size, action_size, goal_dim, target_count):
        super(RewardNet, self).__init__()
        self.action_size = action_size

        input_size = state_size + action_size + goal_dim + target_count
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state, action, goal_delta, target_onehot):
        # action: (batch, 1) 또는 (batch,) → one-hot 변환 필요
        if action.dim() == 2:
            action = action.squeeze(1)

        action_onehot = F.one_hot(action, num_classes=self.action_size).float()
        x = torch.cat([state, action_onehot, goal_delta, target_onehot], dim=1)

        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        return self.out(x)
