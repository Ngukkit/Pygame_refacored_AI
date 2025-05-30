import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from model import DuelingDQNModel, RewardNet
from per import PrioritizedReplayBuffer

class DualReplayAgent:
    def __init__(self, state_size, action_size, goal_indices, target_count, device):
        self.state_size = state_size
        self.action_size = action_size
        self.goal_indices = goal_indices
        self.target_count = target_count
        self.device = device
        self.batch_size = 64
        self.gamma = 0.99
        self.q_values_view = 0

        self.model = DuelingDQNModel(state_size, action_size, target_count).to(device)
        self.target_model = DuelingDQNModel(state_size, action_size, target_count).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)

        self.reward_net = RewardNet(state_size, action_size, len(goal_indices), target_count).to(device)
        self.reward_net_optimizer = optim.Adam(self.reward_net.parameters(), lr=0.0001)

        self.per_buffer = PrioritizedReplayBuffer(100000)
        self.reward_buffer = deque(maxlen=100000)

        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, target_onehot, epsilon=0.1):
        if np.random.rand() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        target_tensor = torch.FloatTensor(np.array(target_onehot)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor, target_tensor)
            self.q_values_view = q_values.detach().cpu()
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done, target_id):
        self.per_buffer.add((state, action, reward, next_state, done, target_id))

        goal_delta = next_state[self.goal_indices] - state[self.goal_indices]
        self.reward_buffer.append((state, action, goal_delta, target_id))

    def train_reward_net(self):
        if len(self.reward_buffer) < self.batch_size:
            return
        samples = random.sample(self.reward_buffer, self.batch_size)
        states, actions, goal_deltas, target_ids = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        goal_deltas = torch.FloatTensor(np.array(goal_deltas)).to(self.device)
        target_onehots = torch.nn.functional.one_hot(torch.LongTensor(target_ids), num_classes=self.target_count).float().to(self.device)

        fake_rewards = torch.ones(self.batch_size, 1).to(self.device)  # 목표 성공 trajectory에서 호출된다고 가정

        pred_rewards = self.reward_net(states, actions, goal_deltas, target_onehots)
        loss = nn.MSELoss()(pred_rewards, fake_rewards)

        self.reward_net_optimizer.zero_grad()
        loss.backward()
        self.reward_net_optimizer.step()

    def replay(self):
        if len(self.per_buffer) < self.batch_size:
            return
        samples, indices, is_weights = self.per_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, target_ids = zip(*samples)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
        target_onehots = torch.nn.functional.one_hot(torch.LongTensor(target_ids), num_classes=self.target_count).float().to(self.device)

        q_values = self.model(states, target_onehots).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_model(next_states, target_onehots).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        td_errors = q_values - target_q
        loss = (is_weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.per_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.train_reward_net()

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'reward_net': self.reward_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'reward_net_optimizer': self.reward_net_optimizer.state_dict(),
            'input_dim': self.state_size,
            'output_dim': self.action_size,
            'target_count': self.target_count,
            'goal_indices': self.goal_indices,  # ✅ 추가
            'epsilon': getattr(self, 'epsilon', None),
            'step': getattr(self, 'total_steps', None),
            'epoch': getattr(self, 'epoch', None),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['target_model'])
        self.reward_net.load_state_dict(checkpoint['reward_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.reward_net_optimizer.load_state_dict(checkpoint['reward_net_optimizer'])

        self.state_size = checkpoint.get('input_dim', self.state_size)
        self.action_size = checkpoint.get('output_dim', self.action_size)
        self.target_count = checkpoint.get('target_count', self.target_count)
        self.goal_indices = checkpoint.get('goal_indices', self.goal_indices)
        self.epsilon = checkpoint.get('epsilon', 1.0)
        self.total_steps = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)

