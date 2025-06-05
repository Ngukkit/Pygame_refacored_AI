
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class DuelingDQNWithLSTM(nn.Module):
    def __init__(self, state_size, action_size, target_count, lstm_hidden=256, lstm_layers=1):
        super(DuelingDQNWithLSTM, self).__init__()
        self.input_size = state_size + target_count  # 상태 + 목표
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.fc1 = nn.Linear(self.input_size, 256)

        # LSTM 입력은 (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True)

        self.fc2 = nn.Linear(lstm_hidden, 128)

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, action_size)
        )

    def forward(self, state_seq, target_onehot_seq, hidden=None):
        """
        state_seq: (batch, seq_len, state_dim)
        target_onehot_seq: (batch, seq_len, target_count)
        hidden: (h0, c0), 각 (lstm_layers, batch, hidden_size)
        """
        x = torch.cat([state_seq, target_onehot_seq], dim=-1)  # (batch, seq_len, input_size)
        x = F.leaky_relu(self.fc1(x))  # (batch, seq_len, 128)
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq_len, lstm_hidden)
        x = F.leaky_relu(self.fc2(lstm_out))  # (batch, seq_len, 128)

        value = self.value_stream(x)         # (batch, seq_len, 1)
        advantage = self.advantage_stream(x) # (batch, seq_len, action_size)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q, hidden


class TrajectoryReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

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
