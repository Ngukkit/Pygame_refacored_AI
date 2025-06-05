
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from per import PrioritizedReplayBuffer
from model_lstm import DuelingDQNWithLSTM, TrajectoryReplayBuffer, RewardNet
# from model import RewardNet  # 기존 1-step RewardNet 그대로 사용

class DualReplayAgentLSTM:
    def __init__(self, state_size, action_size, goal_indices, target_count, device, traj_len=30, lr = 0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.goal_indices = goal_indices
        self.target_count = target_count
        self.q_values_view = None  # Q-values for visualization
        self.device = device
        self.traj_len = traj_len
        self.batch_size = 64
        self.gamma = 0.8  # Discount factor for future rewards

        self.model = DuelingDQNWithLSTM(state_size, action_size, target_count).to(device)
        self.target_model = DuelingDQNWithLSTM(state_size, action_size, target_count).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)

        self.reward_net = RewardNet(state_size, action_size, len(goal_indices), target_count).to(device)
        self.reward_net_optimizer = optim.Adam(self.reward_net.parameters(), lr)

        self.trajectory_buffer = TrajectoryReplayBuffer()
        self.reward_buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember_trajectory(self, trajectory):
        self.trajectory_buffer.add(trajectory)
       
        # reward = trajectory['rewards']
        # if any(r != 0 for r in reward):  # trajectory 내에 delayed reward가 포함되어 있다면
        #     for i in range(len(trajectory['states']) - 1):
        #         s = trajectory['states'][i]
        #         a = trajectory['actions'][i]
        #         ns = trajectory['states'][i + 1]
        #         goal_delta = ns[self.goal_indices] - s[self.goal_indices]
        #         tid = trajectory['target_ids'][i]
        #         sq = trajectory['state_seq'][i]
        #         tq = trajectory['target_seq'][i]
        #         self.reward_buffer.append((s, a, goal_delta, tid,sq, tq))

    def act(self, state_seq_tensor, target_onehot_seq_tensor, epsilon=0.1):
        if np.random.rand() < epsilon:
            return random.randint(0, self.action_size - 1)
        state_seq = state_seq_tensor.to(self.device)  # 이미 (1, seq_len, state_dim) 형태일 경우
        target_seq = target_onehot_seq_tensor.to(self.device)

        with torch.no_grad():
            q_values, _ = self.model(state_seq, target_seq)
            last_q_values = q_values[0, -1] 
            self.q_values_view = last_q_values.cpu().numpy()  # Store for visualization
        return torch.argmax(last_q_values).item()

    def replay_from_trajectory(self):
        if len(self.trajectory_buffer) < self.batch_size:
            return

        batch = self.trajectory_buffer.sample(self.batch_size)
        state_seqs = torch.tensor(np.array([b['states'] for b in batch]), dtype=torch.float32).to(self.device)
        target_seqs = torch.tensor(np.array([b['target_onehots'] for b in batch]), dtype=torch.float32).to(self.device)
        action_seqs = torch.tensor(np.array([b['actions'] for b in batch]), dtype=torch.long).to(self.device)
        reward_seqs = torch.tensor(np.array([b['rewards'] for b in batch]), dtype=torch.float32).to(self.device)
        next_state_seqs = torch.tensor(np.array([b['next_states'] for b in batch]), dtype=torch.float32).to(self.device)
        done_seqs = torch.tensor(np.array([b['dones'] for b in batch]), dtype=torch.float32).to(self.device)
        # 직접 준비한 시퀀스 (LSTM 전용)
        state_seq_tensor = torch.tensor(np.array([b['state_seq'] for b in batch]), dtype=torch.float32).to(self.device)
        target_seq_tensor = torch.tensor(np.array([b['target_seq'] for b in batch]), dtype=torch.float32).to(self.device)

        q_vals, _ = self.model(state_seq_tensor, target_seq_tensor)
        q_taken = q_vals.gather(-1, action_seqs.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_vals, _ = self.target_model(next_state_seqs, target_seqs)
            next_q_max = next_q_vals.max(-1)[0]
            target_q = reward_seqs + self.gamma * next_q_max * (1 - done_seqs)
            
            # === gamma 영향 로그 추가 ===
            # if np.random.rand() < 0.1:  # 1% 확률로만 출력하여 과도한 로그 방지
            #     for i in range(min(3, self.batch_size)):  # 최대 3개 샘플만
            #         print(f"\n[Gamma Test] Sample {i}")
            #         print(f"  reward:     {reward_seqs[i].cpu().numpy()}")
            #         print(f"  next_q_max: {next_q_max[i].cpu().numpy()}")
            #         print(f"  done:       {done_seqs[i].cpu().numpy()}")
            #         print(f"  gamma:      {self.gamma}")
            #         print(f"  target_q:   {target_q[i].cpu().numpy()}")

        loss = nn.MSELoss()(q_taken, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_reward_net(self, beta=0.4):
        if len(self.reward_buffer) < self.batch_size:
            return

        # 샘플링
        samples, indices, weights = self.reward_buffer.sample(self.batch_size)
        states, actions, goal_deltas, target_ids = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        goal_deltas = torch.FloatTensor(np.array(goal_deltas)).to(self.device)
        target_onehots = torch.nn.functional.one_hot(
            torch.LongTensor(target_ids), num_classes=self.target_count
        ).float().to(self.device)

        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 가짜 보상 타깃: 모두 1
        fake_rewards = torch.ones(self.batch_size, 1).to(self.device)

        pred_rewards = self.reward_net(states, actions, goal_deltas, target_onehots)
        loss_per_sample = (pred_rewards - fake_rewards).pow(2)

        # PER 가중치 적용
        weighted_loss = (loss_per_sample * weights).mean()

        self.reward_net_optimizer.zero_grad()
        weighted_loss.backward()
        self.reward_net_optimizer.step()

        # priority 업데이트 (TD-error → reward 예측 오차 사용)
        td_errors = loss_per_sample.detach().cpu().numpy().flatten()
        self.reward_buffer.update_priorities(indices, td_errors)


    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'reward_net': self.reward_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'reward_net_optimizer': self.reward_net_optimizer.state_dict(),
            'goal_indices': self.goal_indices,
            'traj_len': self.traj_len
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['target_model'])
        self.reward_net.load_state_dict(checkpoint['reward_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.reward_net_optimizer.load_state_dict(checkpoint['reward_net_optimizer'])
        self.goal_indices = checkpoint['goal_indices']
        self.traj_len = checkpoint['traj_len']
