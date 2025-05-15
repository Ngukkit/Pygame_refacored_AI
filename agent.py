import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.norm1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.norm3 = nn.LayerNorm(64)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, 32)
        self.norm4 = nn.LayerNorm(32)
        self.drop4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(32, 16)
        self.norm5 = nn.LayerNorm(16)
        self.drop5 = nn.Dropout(0.2)

        self.out = nn.Linear(16, action_size)

    def forward(self, x, return_hidden=False):
        x = self.drop1(F.leaky_relu(self.norm1(self.fc1(x))))
        h1 = x
        x = self.drop2(F.leaky_relu(self.norm2(self.fc2(x))))
        h2 = x
        x = self.drop3(F.leaky_relu(self.norm3(self.fc3(x))))
        h3 = x
        x = self.drop4(F.leaky_relu(self.norm4(self.fc4(x))))
        h4 = x
        x = self.drop5(F.leaky_relu(self.norm5(self.fc5(x))))
        h5 = x
        out = self.out(x)

        if return_hidden:
            return out, h1, h2, h3, h4, h5
        else:
            return out

class RewardNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size + state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action_onehot, state_delta):
        x = torch.cat((state, action_onehot, state_delta), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # return torch.sigmoid(self.fc3(x))  # 0~1 보상 예측용
        return self.fc3(x)  # ⛔ sigmoid 제거!


class StatePredictionNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(StatePredictionNet, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, state_size)

    def forward(self, state, action_onehot):
        x = torch.cat((state, action_onehot), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # return torch.sigmoid(self.fc3(x))  # 예측 상태도 0~1 정규화
        return self.fc3(x)  # ⛔ sigmoid 제거!

# class StateExplorationReward:
#     def __init__(self, max_reward_steps=500, exploration_boost=0.020):
#         self.max_reward_steps = max_reward_steps  # 보상을 받지 못한 최대 시간
#         self.exploration_boost = exploration_boost  # 탐색 보상 증가량
#         self.no_reward_steps = 0                    # 보상을 받지 못한 시간 추적
#         self.total_steps = 0                             # 전체 스텝 수

#     def compute_reward(self,reward,step,lstep, epsilon):

#         self.lsteps = lstep       
#         self.step = step
         
#         if reward > 0:
#             print(f"reward = {reward}, total_step = {self.lsteps}")
            

#         # 마지막 보상 이후 500 스텝 이상 아무 보상이 없었을 때
#         if self.step - self.lsteps  == self.max_reward_steps:
#             print(f"total steps: {self.step - self.lsteps} 스텝 epsilon UP ")
#             epsilon = min(epsilon + self.exploration_boost, 1)

#         return epsilon
        
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma

        self.model = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.reward_model = RewardNet(state_size, action_size)
        self.prediction_model = StatePredictionNet(state_size, action_size)

        self.update_target_model()
        
        # self.diversity_rewarder = StateDiversityReward(history_size=20, min_diversity_threshold=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.reward_model.to(self.device)
        self.prediction_model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=lr)
        # self.prediction_optimizer = optim.Adam(self.prediction_model.parameters(), lr=lr)
        
        self.criterion = nn.MSELoss()
        self.contrastive_criterion = nn.L1Loss()

        # self.exploration_rewarder = StateExplorationReward()
        self.last_reward = None
        # self.replay_index = 0  # 반복 위치를 저장할 변수
        self.q_valuefordp = 0
        self.epsilon = 1.0
        self.epsilon_min = 0
        self.epsilon_max = 0.9
        self.update_counter = 0
        self.epsilon_decay = 0.00001  # epsilon 감소 비율
        self.reward_ema = 0.0
        self.reward_threshold = 0.05 # 보상 변화 임계값
        self.target_update_frequency = 10
    
        # self.simulated_action = True
        # self.simulated_confidence = 0.0
        # self.simulated_decay = 0.95  # 점점 감소

        self.recent_actions = deque(maxlen=10)  # 최근 행동 기록
        self.epsilon_boost = 0.8         # 복구시 epsilon 증가량
        self.nstep_buffer = deque(maxlen=10)  # N=5, 추적할 최대 길이
        self.lambda_decay = 0.9              # λ 값

        
    # def is_action_available(self, action, state_tensor):
    #     """
    #     기본 행동 유효성 검사 함수
    #     예: 스킬 개수와 상태에 따른 행동 제한
    #     """
    #     skill_state_index = 5
    #     if action <= 4:  # 기본적인 행동들: 이동, 점프, 공격 등
    #         return True
    #     skillget_estimated = int(state_tensor[0][skill_state_index].item())  # 상태 내 인덱스에서 추출
    #     # 스킬 개수에 따른 행동 제한
    #     if action == 5 and skillget_estimated >= 1:
    #         return True
    #     elif action == 6 and skillget_estimated >= 2:
    #         return True
    #     elif action == 7 and skillget_estimated >= 3:
    #         return True
    #     elif action == 8 and skillget_estimated >= 4:
    #         return True
    #     elif action == 9 and skillget_estimated >= 5:
    #         return True
        
    #     return False  # 그 외에는 유효하지 않음

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def calculate_reward(self,predicted_reward, masked_state_delta, lambda_factor=0.05):
        # 상태 변화량의 norm을 계산 (Euclidean norm)
        delta_norm = torch.norm(masked_state_delta, p=2).item()
        
        # 보상 계산
        pre_reward = predicted_reward + lambda_factor * delta_norm
        
        return pre_reward
        
    def act(self, state, reward=None, use_reward_simulation=False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # # 보상 기반 탐색
        # self.epsilon = self.exploration_rewarder.compute_reward(reward,step,lstep, epsilon=self.epsilon)
        
        if np.random.rand() <= self.epsilon:
            # if np.random.rand() <= 0.01:
            #     print(f"random", end=" ")
            return random.randrange(self.action_size)
        else:
            if use_reward_simulation:
                if self.device.type == "cuda":

                    # 🔹 GPU 병렬 연산 버전
                    state_expanded = state_tensor.repeat(self.action_size, 1)  # (10, state_size)
                    onehot_actions = torch.eye(self.action_size).to(self.device)  # (10, action_size)
                    predicted_next_states = self.prediction_model(state_expanded, onehot_actions)  # (10, state_size)
                    state_deltas = predicted_next_states - state_expanded

                    mask = torch.ones_like(state_deltas)
                    mask[:, 0] = 0  # x 위치
                    mask[:, 1] = 0  # 점프 여부
                    mask[:, 3] = 0  # 방향
                    masked_state_deltas = torch.clamp(state_deltas * mask, min=-0.3, max=0.3)

                    predicted_rewards = self.reward_model(state_expanded, onehot_actions, masked_state_deltas).squeeze(1)
                    delta_norms = torch.norm(masked_state_deltas, dim=1)
                    combined_rewards = predicted_rewards + 0.1 * delta_norms
                    simulated_rewards = combined_rewards.detach().cpu().numpy()

                    q_values = self.model(state_tensor).detach().squeeze(0).clamp(min=0).log1p()
                    alpha = 0.1 + 0.9 * self.epsilon
                    hybrid_values = alpha * combined_rewards + (1 - alpha) * q_values 
                    action = int(torch.argmax(hybrid_values).item())
                    
                    if np.random.rand() <= 0.1:
                        print(f"[GPU] ep:{self.epsilon:.1f} maxQ:{torch.max(hybrid_values).item():.3f}, q_vals:{torch.max(q_values).item():.3f}, pre:{np.max(simulated_rewards):.3f}\n")


                else:
                    # 🔹 CPU fallback: 기존 for 루프 방식
                    # 보상 예측 기반 테스트 시뮬레이션
                    q_values = self.model(state_tensor).detach().cpu().numpy()
                    q_values = np.clip(q_values, a_min=0.0, a_max=None)  # 음수를 0으로 처리
                    q_values = np.log1p(q_values)
                    
                    n_step = 10
                    gamma = 0.99
                    # simulated_rewards = [-1.0] * self.action_size  # 전체 길이 맞춰 초기화
                    simulated_rewards = [] 
                            
                    # 행동 시뮬레이션
                    for action in range(self.action_size):
                        cumulative_reward = 0.0
                        discount = 1.0
                        current_state = state_tensor.clone()
                        
                        # if not self.is_action_available(action, state_tensor):  # 예: skillget 부족 등
                        #     simulated_rewards.append(-1.0)
                        #     continue
                        
                        for step in range(n_step):
                            # 1. One-hot 인코딩된 행동
                            onehot_action = torch.nn.functional.one_hot(
                                torch.tensor([action]), num_classes=self.action_size
                            ).float().to(self.device)

                            # 2. 다음 상태 예측
                            predicted_next_state = self.prediction_model(current_state, onehot_action)

                            # 3. 상태 변화량 계산
                            state_delta = predicted_next_state - current_state
                            
                            # 🎯 0번, 1번 상태 변화는 무시 (masking)
                            mask = torch.ones_like(state_delta)
                            mask[0][0] = 0  # x 위치
                            mask[0][1] = 0  # 점프 여부
                            mask[0][3] = 0  # 방향
                            masked_state_delta = state_delta * mask

                            # masked_state_delta 스케일링 예시
                            masked_state_delta = torch.clamp(masked_state_delta, min=-0.3, max=0.3)

                            # 4. 보상 예측
                            predicted_reward = self.reward_model(current_state, onehot_action, masked_state_delta)

                            # 5. 누적 보상 계산 (할인 적용)

                            cumulative_reward += self.calculate_reward(predicted_reward, masked_state_delta, lambda_factor=0.1)
                            if reward is not None:
                                weighted_reward = 0.2 * cumulative_reward + 0.8 * (reward / 20)# 보상 가중치 조정
                                cumulative_reward += weighted_reward.item()
                            # cumulative_reward += discount * predicted_reward.item()
                            discount *= gamma
                            
                            # 다음 상태로 이동
                            current_state = predicted_next_state.detach()  # detach 중요
                        simulated_rewards.append(cumulative_reward.item())  # cumulative_reward.item()으로 변환

                    # 🎯 혼합: Q값과 예측 보상의 가중 평균
                    # alpha = max(0.1, self.epsilon )  # 가중치 (0=Q만, 1=예측simulated만)
                    q_values = q_values.squeeze()  # (1, 10) → (10,)
                    alpha = 0.1 + 0.9 * self.epsilon  
                    hybrid_values = alpha * np.array(simulated_rewards) + (1 - alpha) * (q_values)
                    
                    if np.random.rand() <= 0.1:
                        print(f"ep:{self.epsilon:.1f} maxQ:{np.max(hybrid_values):.3f}, q_vals:{np.max(q_values)},pre:{np.max(simulated_rewards)}\n")

                    # 가장 보상이 클 것으로 예측되는 행동 선택
                    action = int(np.argmax(hybrid_values))
                    # if np.array(simulated_rewards).max() >= 0.8 and not reward < 0:
                    # # if reward < 0:
                    #     self.simulated_action = action  # 시뮬레이션 행동 유지
            else:
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
                    self.q_valuefordp = np.max(q_values)
                    # if np.random.rand() <= 0.1:
                    #     print(f"ep:{self.epsilon} q_vals:{np.max(q_values)}\n",end = '')
                return int(np.argmax(q_values))
                          
         # 최근 행동을 기록       
        self.recent_actions.append(action)
        self.simulated_confidence = min(1.0, np.max(simulated_rewards))  # 5.0은 normalization factor
        return action

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()
        # state_delta = np.array(next_state) - np.array(state)
        # state_delta = next_state - state  # 상태 변화량 계산
        # transition = (state, action, reward, next_state, done, state_delta)
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)
        
        # self.nstep_buffer.append(transition)

        # # N개가 쌓이면 trajectory를 memory에 저장
        # if len(self.nstep_buffer) == self.nstep_buffer.maxlen:
        #     self.memory.append(list(self.nstep_buffer))


    def replay(self, batch_size=50):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        batch = np.array(minibatch, dtype=object)

        # 🔹 [1] RewardNet, PredictionNet, Contrastive 학습은 기존 방식 유지
        # flat_batch = [t for traj in minibatch for t in traj]  # trajectory 내 transition 평탄화
        # batch = np.array(flat_batch, dtype=object)
        

        states = torch.tensor(np.stack(batch[:, 0].tolist()), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(batch[:, 1].tolist()), dtype=torch.long).unsqueeze(1).to(self.device)
        # onehot_actions = torch.nn.functional.one_hot(actions.squeeze(), num_classes=self.action_size).float().to(self.device)
        rewards = torch.tensor(np.stack(batch[:, 2].tolist()), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.stack(batch[:, 3].tolist()), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.stack(batch[:, 4].tolist()), dtype=torch.float32).unsqueeze(1).to(self.device)
        # state_deltas = torch.tensor(np.stack(batch[:, 5].tolist()), dtype=torch.float32).to(self.device)

        
        # 현재 Q값
        q_values = self.model(states).gather(1, actions)

        # 타겟 Q값
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
        # 손실 계산 및 업데이트
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # 1. RewardNet 학습
        # predicted_rewards = self.reward_model(states, onehot_actions, state_deltas)
        # reward_loss = self.criterion(predicted_rewards, rewards)


        # # 2. StatePredictionNet 학습
        # predicted_next_states = self.prediction_model(states, onehot_actions)
        # prediction_loss = self.criterion(predicted_next_states, next_states)

        # # 3. Contrastive 학습: 보상이 없는 transition은 학습 weight를 낮춤
        # with torch.no_grad():
        #     prediction_error = (predicted_next_states - next_states).pow(2).mean(dim=1, keepdim=True)  # [batch, 1]
        #     prediction_error = prediction_error / (prediction_error.max() + 1e-6)

        # # contrastive_loss = prediction_error * reward 또는 그 반대 depending on 의미 설정
        # contrastive_loss = self.contrastive_criterion(prediction_error, rewards)  # 의미: 보상이 있을수록 예측이 어려워야

        # # 4. RewardNet 학습에 contrastive loss 반영
        # total_reward_loss = reward_loss + 0.1 * contrastive_loss
        # self.reward_optimizer.zero_grad()
        # total_reward_loss.backward()
        # self.reward_optimizer.step()

        # # 5. PredictionNet 학습
        # self.prediction_optimizer.zero_grad()
        # prediction_loss.backward()
        # self.prediction_optimizer.step()

        # # 🔹 [2] TD(λ) 기반 Q-network 학습
        # lambda_decay = self.lambda_decay  # 예: 0.9

        # for trajectory in minibatch:  # trajectory = N-step list of transitions
        #     cumulative_td = 0.0

        #     for i, (s, a, r, s_next, done, _) in enumerate(trajectory):
        #         state_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
        #         next_state_tensor = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0).to(self.device)
        #         action_tensor = torch.tensor([[a]]).to(self.device)

                # # 현재 Q값
                # q_val = self.model(state_tensor).gather(1, action_tensor)
                
                # # 타겟 Q값
                # with torch.no_grad():
                #     target_q = self.target_model(next_state_tensor).max(1, keepdim=True)[0]
                #     td_target = torch.tensor([[r]], device=self.device) + (1 - done) * self.gamma * target_q
                
                # td_error = td_target - q_val
                # weight = (lambda_decay ** i)
                # cumulative_td += weight * td_error.item()
                # if done:
                #     break

            

            # trajectory의 첫 state-action만 업데이트 (가중 누적 TD-error 반영)
            # 전체 trajectory에 대해 첫 transition을 사용
            # s0, a0, r0, s1, done, _ = trajectory[0]
            # s0_tensor = torch.tensor(s0, dtype=torch.float32).unsqueeze(0).to(self.device)
            # a0_tensor = torch.tensor([[a0]]).to(self.device)

            # with torch.no_grad():
            #     max_q_next = self.target_model(torch.tensor(s1, dtype=torch.float32).unsqueeze(0).to(self.device)).max(1, keepdim=True)[0]
            #     td_target = torch.tensor([[r0]], device=self.device) + (1 - done) * self.gamma * max_q_next

            # q_val = self.model(s0_tensor).gather(1, a0_tensor)
            # loss = self.criterion(q_val, td_target)

            # # 보상 예측 확인
            # print(f"[replay] predicted_reward.mean = {q_val0.mean().item():.4f}")
            # print(f"[replay] loss = {loss.item():.4f}")
            # print(f"[replay] contrastive_loss = {contrastive_loss.item():.4f}")
            # print(f"[replay] prediction_loss = {prediction_loss.item():.4f}")

            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            # print(f"[replay] Q-loss = {loss.item():.4f}")

        # #보상 변화에 따른 epsilon 증가
        # reward_mean = rewards.mean().item()
        # self.reward_ema = 0.9 * self.reward_ema + 0.1 * reward_mean  # 초기값 0
        # reward_delta = reward_mean - self.reward_ema

        # if reward_delta < -self.reward_threshold:  # 최근 평균보다 급감했을 때
        #     self.epsilon = min(self.epsilon + self.epsilon_boost, 0.8)  # epsilon을 올림
        #     print(f"epsilon 증가: {self.epsilon:.2f}")

        # self.last_reward = rewards.mean().item()
        
        # Epsilon 감소      
        # cycle = 10000  
        # self.epsilon = max(self.epsilon_min, 0.5 * (1 + np.cos(2 * np.pi * (step % cycle) / cycle)))
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # self.update_counter += 1
        # if self.update_counter % self.target_update_frequency == 0:
        #     self.update_target_model()
            
    def log_rewardnet_outputs(self, log_path="rewardnet_log.npy", sample_size=200):
        if len(self.memory) < sample_size:
            return
        
        # 최근 trajectory 평탄화
        flat_batch = [t for traj in list(self.memory)[-sample_size:] for t in traj]
        batch = np.array(flat_batch, dtype=object)

        states = torch.tensor(np.stack(batch[:, 0].tolist()), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(batch[:, 1].tolist()), dtype=torch.long).unsqueeze(1).to(self.device)
        onehot_actions = torch.nn.functional.one_hot(actions.squeeze(), num_classes=self.action_size).float().to(self.device)
        state_deltas = torch.tensor(np.stack(batch[:, 5].tolist()), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predicted_rewards = self.reward_model(states, onehot_actions, state_deltas).cpu().numpy()

        np.save(log_path, {
            "states": states.cpu().numpy(),
            "actions": actions.cpu().numpy(),
            "deltas": state_deltas.cpu().numpy(),
            "predicted_rewards": predicted_rewards
    })
    
            
    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            # 'prediction_model': self.prediction_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'reward_optimizer': self.reward_optimizer.state_dict(),
            # 'prediction_optimizer': self.prediction_optimizer.state_dict(),
            'epsilon': self.epsilon
            # 'memory': [list(traj) for traj in self.memory],  # trajectory 리스트 저장
            # 'nstep_buffer': list(self.nstep_buffer),         # 현재 쌓인 n-step 버퍼도 저장
            # 'update_counter': self.update_counter
        }, path)


    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        # self.prediction_model.load_state_dict(checkpoint['prediction_model'])
        self.optimizer.load_state_dict(checkpoint.get('optimizer', self.optimizer.state_dict()))
        # self.reward_optimizer.load_state_dict(checkpoint.get('reward_optimizer', self.reward_optimizer.state_dict()))
        # self.prediction_optimizer.load_state_dict(checkpoint.get('prediction_optimizer', self.prediction_optimizer.state_dict()))
        self.epsilon = checkpoint.get('epsilon', self.epsilon)

        # # ✅ memory와 nstep_buffer 복원
        # if 'memory' in checkpoint:
        #     self.memory = deque([list(t) for t in checkpoint['memory']], maxlen=8000)

        # if 'nstep_buffer' in checkpoint:
        #     self.nstep_buffer = deque([tuple(t) for t in checkpoint['nstep_buffer']], maxlen=self.nstep_buffer.maxlen)

        # self.update_counter = checkpoint.get('update_counter', 0)

        self.update_target_model()
