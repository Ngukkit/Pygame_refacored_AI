import os
import argparse
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 예시용 간단한 모델 정의
class DQN(nn.Module):
    def __init__(self, state_size, action_size, target_count, lstm_hidden=256, lstm_layers=1):
        super(DQN, self).__init__()
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

# 모델 로드 함수
def load_model(model_path, input_dim, output_dim, target_count=8):
    model =DQN(state_size=input_dim - target_count, action_size=output_dim, target_count=target_count)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, input_dim, output_dim

def visualize_q_over_time(log_files):
    for file in log_files:
        with open(file, "r") as f:
            lines = f.readlines()

        all_q_values = []

        for line in lines:
            try:
                if line.strip() == "":
                    continue  # 빈 줄 무시
                entry = json.loads(line)
                q_values = entry.get("q_values")
                if isinstance(q_values, list) and len(q_values) > 0:
                    inner = q_values[0]
                    if isinstance(inner, list) and len(inner) == 6:
                        all_q_values.append(np.array(inner))
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error in {file}: {e}")
                continue

        if not all_q_values:
            print(f"⚠️ No valid Q-values found in {file}")
            continue
        # for i, q in enumerate(all_q_values):
        #     print(f"{i}: {type(q)}, shape={np.array(q).shape if hasattr(q, '__len__') else 'scalar'}")
        
        data = np.array(all_q_values)  # shape: (steps, actions)

        # plt.figure(figsize=(8,6))
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=f'Action {i}')
        plt.title(f"Q-values over time ({file})")
        plt.xlabel("Step")
        plt.ylabel("Q-value")
        plt.legend()
        plt.grid(True)
        # plt.show()
        # plt.savefig(f"q_over_time_{file.replace('.json','')}.png")
        # plt.close()

def visualize_n_state_time(log_files):
    for npy_file in log_files:
        print(f"Loading {npy_file}...")
        data = np.load(npy_file, allow_pickle=True).item()
        states = data["states"]
        deltas = data["deltas"]
        rewards = data["predicted_rewards"].flatten()

        # PCA로 시각화용 2D로 축소
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        state_features = pca.fit_transform(np.hstack([states, deltas]))

        # 산점도
        # plt.figure(figsize=(8,6))
        sc = plt.scatter(state_features[:,0], state_features[:,1], c=rewards, cmap='viridis', s=30)
        plt.colorbar(sc, label="Predicted Reward")
        plt.title(f"RewardNet Prediction: {npy_file}")
        plt.xlabel("state by Step")
        plt.ylabel("Q-value predicted")
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"rewardnet_{npy_file.replace('.npy', '')}.png")
        plt.close()


# 상태에 따른 Q값 시각화
# 각 행동(Action) Q값의 상태 의존성 확인
#     상태의 변화에 따라 어떤 행동의 Q값이 커지거나 작아지는지 확인할 수 있음

# 결정 경계(Decision Boundary) 추정
#     예: 상태가 0.3 이하일 땐 Action 0이 Q값이 가장 높지만, 그 이상에선 Action 2가 더 높다면
#     → 그 지점이 행동 결정 경계

# 정책의 민감도 확인
#     상태가 조금만 바뀌어도 Q값이 급격히 바뀐다면 → 정책이 매우 민감
#     반대로 Q값이 평평하면 → 학습이 잘 안 되었을 가능성 있음

# Q-value saturation 또는 collapse 현상 확인
#     모든 Q값이 거의 일정하거나 0 근처에 몰려 있다면 → 네트워크가 학습되지 않았거나 gradient vanishing 가능성

# 탐색(exploration)이 필요한 영역 확인
#     상태 특정 구간에서 Q값이 다 작거나 엇비슷하면 → 해당 영역에서 학습 데이터 부족했을 수 있음

def visualize_q_across_states(model, input_dim, output_dim):
    state_dim = 28
    target_dim = 8
    num_states = 1000  # 시각화할 상태 수
    states = np.random.rand(num_states, state_dim).astype(np.float32)
    
    # 랜덤하게 target_id 부여
    target_ids = np.random.randint(0, target_dim, size=(num_states,))
    target_onehots = np.eye(target_dim)[target_ids].astype(np.float32)

    # 두 개 합쳐서 최종 입력 (28 + 8 = 36)
    states_tensor = torch.tensor(states)
    targets_tensor = torch.tensor(target_onehots)
    with torch.no_grad():
        q_values = model(states_tensor, targets_tensor).numpy()

    # plt.figure(figsize=(8, 6))
    for i in range(output_dim):
        plt.plot(states[:, 0], q_values[:, i], label=f'Action {i}')
    plt.title("Q-values across state space")
    plt.xlabel("State (normalized)")
    plt.ylabel("Q-value")
    plt.legend()
    plt.grid(True)


def visualize_q_and_policy_3d(model, input_dim, output_dim, method='tsne', num_states=1000):
    state_dim = 28
    target_dim = 8
    num_states = 1000  # 시각화할 상태 수
    states = np.random.rand(num_states, state_dim).astype(np.float32)
    
    # 랜덤하게 target_id 부여
    target_ids = np.random.randint(0, target_dim, size=(num_states,))
    target_onehots = np.eye(target_dim)[target_ids].astype(np.float32)

    # 두 개 합쳐서 최종 입력 (28 + 8 = 36)
    full_inputs = np.concatenate([states, target_onehots], axis=1)  
    with torch.no_grad():
        q_values = model(torch.tensor(full_inputs)).numpy()
        max_q = np.max(q_values, axis=1)
        actions = np.argmax(q_values, axis=1)

    if method == 'pca':
        reducer = PCA(n_components=3)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=30, max_iter=1000)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    reduced = reducer.fit_transform(states)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    p1 = ax1.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=max_q, cmap='viridis', s=20)
    ax1.set_title(f"Q-value Heatmap (3D {method.upper()})")
    ax1.set_xlabel("Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.set_zlabel("Dim 3")
    fig.colorbar(p1, ax=ax1, shrink=0.6, label="Max Q-value")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    cmap = plt.colormaps['tab10']
    colors = cmap(np.linspace(0, 1, output_dim))
    p2 = ax2.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=colors[actions], s=20)
    ax2.set_title(f"Policy Map (3D {method.upper()})")
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.set_zlabel("Dim 3")
    fig.colorbar(p2, ax=ax2, shrink=0.6, label="Argmax Action")

    plt.tight_layout()
    plt.show()


    
def visualize_q_pca_tsne(model, input_dim, method='pca', num_states=1000):
    state_dim = 28
    target_dim = 8
    num_states = 1000  # 시각화할 상태 수
    states = np.random.rand(num_states, state_dim).astype(np.float32)
    
    # 랜덤하게 target_id 부여
    target_ids = np.random.randint(0, target_dim, size=(num_states,))
    target_onehots = np.eye(target_dim)[target_ids].astype(np.float32)

    # 두 개 합쳐서 최종 입력 (28 + 8 = 36)
    states_tensor = torch.tensor(states)
    targets_tensor = torch.tensor(target_onehots) 
    with torch.no_grad():
        q_values = model(states_tensor, targets_tensor).numpy()
        max_q = np.max(q_values, axis=1)

    reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, perplexity=30)
    reduced = reducer.fit_transform(states)

    # plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=max_q, cmap='viridis', s=15)
    plt.colorbar(label="Max Q-value")
    plt.title(f"Q-value Heatmap ({method.upper()})")
    # plt.legend()
    plt.grid(True)


def visualize_policy_boundary(model, input_dim, method='pca', num_states=1000):
    state_dim = 28
    target_dim = 8
    num_states = 1000  # 시각화할 상태 수
    states = np.random.rand(num_states, state_dim).astype(np.float32)
    
    # 랜덤하게 target_id 부여
    target_ids = np.random.randint(0, target_dim, size=(num_states,))
    target_onehots = np.eye(target_dim)[target_ids].astype(np.float32)

    # 두 개 합쳐서 최종 입력 (28 + 8 = 36)
    states_tensor = torch.tensor(states)
    targets_tensor = torch.tensor(target_onehots) 
    
    with torch.no_grad():
        q_values = model(states_tensor, targets_tensor).numpy()
        actions = np.argmax(q_values, axis=1)

    reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, perplexity=30)
    reduced = reducer.fit_transform(states)

    # plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=actions, cmap='tab10', s=15)
    plt.colorbar(label="Argmax Action")
    plt.title(f"Argmax Policy Map ({method.upper()})")
    # plt.legend()
    plt.grid(True)


def visualize_q_by_state_dims(model, input_dim, output_dim, dims=[2,38,67,70], num_states=1000):
    state_dim = 28
    target_dim = 8
    num_states = 1000  # 시각화할 상태 수
    states = np.random.rand(num_states, state_dim).astype(np.float32)
    
    # 랜덤하게 target_id 부여
    target_ids = np.random.randint(0, target_dim, size=(num_states,))
    target_onehots = np.eye(target_dim)[target_ids].astype(np.float32)

    # 두 개 합쳐서 최종 입력 (28 + 8 = 36)
    states_tensor = torch.tensor(states)
    targets_tensor = torch.tensor(target_onehots)
    
    with torch.no_grad():
        q_values = model(states_tensor, targets_tensor).numpy()
    aa = 1
    for d in dims:
        aa += 1
        plt.subplot(2, 3, aa)  # 2x2 그리드에서 두 번째 위치
        for i in range(output_dim):
            plt.scatter(states[:, d], q_values[:, i], label=f'Action {i}', s=10, alpha=0.5)
        plt.title(f"Q-values vs State Dimension {d}")
        plt.xlabel(f"State Dimension {d}")
        plt.ylabel("Q-value")
        plt.legend()
        plt.grid(True)
        
def plot_hidden_tsne(model, input_dim):
    state_dim = 28
    target_dim = 8
    num_samples = 250

    state_part = torch.rand(num_samples, state_dim) * 10 - 5
    target_ids = torch.randint(0, target_dim, (num_samples,))
    target_onehot = F.one_hot(target_ids, num_classes=target_dim).float()

    x = torch.cat([state_part, target_onehot], dim=1)
    states = state_part
    targets = target_onehot
    
    with torch.no_grad():
        _, h1, h2, h3 = model(states, targets, return_hidden=True)
        hidden = torch.cat([h1, h2, h3], dim=1)  # (500, 256+128+64+32+16=496)

    hidden_np = hidden.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=20, random_state=0)
    hidden_emb = tsne.fit_transform(hidden_np)

    plt.scatter(hidden_emb[:, 0], hidden_emb[:, 1], c=x[:, 0].numpy(), cmap='viridis')
    plt.title("Hidden Activation (t-SNE)")
    plt.colorbar(label="Input x[:, 0]")


def plot_piecewise_structure(model, input_dim):

    num_samples = 1000
    state_dim = 28
    target_dim = 8

    x = torch.linspace(-5, 5, num_samples).unsqueeze(1).repeat(1, input_dim)
    targets = torch.zeros((x.shape[0], target_dim))  # or 무작위 one-hot
    states = x[:, :state_dim]

    with torch.no_grad():
        y = model(states, targets)

    diffs = torch.diff(y, dim=0).abs()
    sharp_changes = (diffs > 0.05).any(dim=1)  # any output dimension exceeds threshold

    for i in range(y.shape[1]):
        plt.plot(x[:, 0].numpy(), y[:, i].numpy(), label=f'Model Output dim {i}')

    plt.plot(x[:-1, 0][sharp_changes].numpy(), y[:-1, 0][sharp_changes].numpy(), 'ro', label='Breakpoint (dim 0)')

    plt.title("Piecewise Linear Approximation with LeakyReLU")
    plt.legend()
    plt.grid(True)


def plot_function_slope(model, input_dim):

    num_samples = 500
    state_dim = 28
    target_dim = 8
    
    x = torch.linspace(-5, 5, num_samples).unsqueeze(1).repeat(1, input_dim)
    
    # 상태만 추출 후 requires_grad 활성화
    states = x[:, :state_dim].clone().detach().requires_grad_(True)
    
    targets = torch.zeros((x.shape[0], target_dim))  # or 무작위 one-hot

    y = model(states, targets)  # (num_samples, output_dim)
    grads = torch.autograd.grad(outputs=y.sum(), inputs=states, create_graph=True)[0]  # (num_samples, input_dim)

    for i in range(min(20, grads.shape[1])):
        plt.plot(x[:, i].detach().numpy(), grads[:, i].detach().numpy(), label=f'dy/dx dim {i}')

    plt.title("Function Slope w.r.t State Inputs")
    plt.xlabel("State Value")
    plt.ylabel("Gradient dy/dx")
    plt.grid(True)
    plt.legend()


def plot_relu_regions(model, input_dim):

    num_samples = 250
    state_dim = 28
    target_dim = 8

    x = torch.linspace(-5, 5, num_samples).unsqueeze(1).repeat(1, input_dim)
    states = x[:, :state_dim]
    targets = torch.zeros((x.shape[0], target_dim))  # or 무작위 one-hot


    with torch.no_grad():
        out, h1, h2, h3 = model(states, targets, return_hidden=True)
        hidden_last = h3  # 마지막 hidden layer (batch, 16)

    for i in range(hidden_last.shape[1]):
        on_region = (hidden_last[:, i] > 0).int()
        plt.plot(x[:, 0].numpy(), on_region.numpy() * (i + 1), label=f'Neuron {i}')

    plt.title("ReLU activation area (LeakyReLU with not ReLU warning)")
    plt.xlabel("x")
    plt.ylabel("activation on/off")
    plt.legend()
    plt.grid(True)


# 시간에 따른 Q값 변화 시각화


 # 상태별 argmax 행동 시각화 
#   같은 상태 차원에서 어떤 행동을 더 많이 선택하는지
#     특정 상태 구간에서 항상 같은 행동을 하는지,
#     혹은 행동 선택이 다양하게 분포되는지

# 결정 경계 (decision boundary) 확인
#     예: 상태값이 0.3 이상이면 항상 action 2를 선택하고, 이하면 action 1을 선택한다면 → 경계가 있다는 의미

# 정책이 편향되어 있는지 확인
#     예: 항상 같은 action만 argmax로 선택된다면, policy가 덜 학습되었거나 Q값 분포가 왜곡되었을 수 있음

# 입력 상태의 어떤 차원이 행동 결정에 큰 영향을 주는지 파악
#     현재 코드는 states[:, 0]만 x축으로 보고 있음
#     고차원 상태의 경우, PCA/TSNE 등을 통해 차원 축소해서 2D로 시각화하는 게 더 유용함      

def visualize_argmax_actions(model, input_dim):
    state_dim = 28
    target_dim = 8

    # 2D 그리드 생성 (100x100 = 10000 points)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(x, y)

    states_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)  # shape (10000, 2)
    states_full = np.zeros((states_2d.shape[0], state_dim), dtype=np.float32)
    states_full[:, 0:2] = states_2d  # 앞 2차원만 채움

    # targets: 하나의 고정된 target_id를 one-hot으로 사용
    fixed_target_id = 0
    targets = np.eye(target_dim, dtype=np.float32)[fixed_target_id]  # (target_dim,)
    targets = np.tile(targets, (states_full.shape[0], 1))  # (10000, target_dim)

    states_tensor = torch.tensor(states_full)
    targets_tensor = torch.tensor(targets)

    with torch.no_grad():
        q_values = model(states_tensor, targets_tensor)  # (10000, output_dim)
        actions = torch.argmax(q_values, dim=1).cpu().numpy()

    plt.contourf(xx, yy, actions.reshape(xx.shape), levels=np.arange(actions.max() + 2) - 0.5, cmap='tab10')
    plt.colorbar()
    plt.title("Argmax Action Decision Boundary")
    plt.xlabel("State Dimension 0")
    plt.ylabel("State Dimension 1")

#-----------------------------------------------------------------------------#
def visualize_target_id_effect(model, base_state, num_targets=8):
    """
    base_state: numpy array, 상태 벡터에서 target_id 부분을 제외한 기본 상태 (길이 = input_dim - num_targets)
    num_targets: target_id one-hot 길이 (예: 8)
    """
    states = np.tile(base_state, (num_targets, 1))  # shape: (num_targets, state_dim)
    targets = np.eye(num_targets, dtype=np.float32)  # shape: (num_targets, num_targets)

    states_tensor = torch.tensor(states, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    with torch.no_grad():
        q_values = model(states_tensor, targets_tensor)  # shape: (num_targets, action_size)
        q_values = q_values.numpy()
        actions = np.argmax(q_values, axis=1)
        max_q = np.max(q_values, axis=1)

    plt.subplot(2, 3, 5)  # 2x3 그리드에서 다섯 번째 위치
    plt.title("Max Q-value vs target_id")
    plt.bar(range(num_targets), max_q)
    plt.xlabel("target_id")
    plt.ylabel("Max Q-value")

    plt.subplot(2, 3, 6)  # 2x3 그리드에서 다섯 번째 위치
    plt.title("Selected Action vs target_id")
    plt.bar(range(num_targets), actions)
    plt.xlabel("target_id")
    plt.ylabel("Action Index")



def visualize_action_softmax_distribution(model, input_dim, fixed_state=None):
    """
    fixed_state: numpy array or None
    만약 fixed_state가 주어지면 해당 상태에 대한 softmax action 확률 분포를 출력
    아니면 랜덤 상태 하나에 대해 출력
    """
    state_dim = 28
    target_dim = 8
    
    if fixed_state is None:
        state = np.random.rand(state_dim).astype(np.float32)
    else:
        if len(fixed_state) != state_dim:
            raise ValueError(f"fixed_state 길이는 {state_dim}이어야 함.")
        state = fixed_state.astype(np.float32)
        # 무작위 target_id 부여 및 one-hot 인코딩
        
    target_id = np.random.randint(0, target_dim)
    target_onehot = np.eye(target_dim)[target_id].astype(np.float32)

    # 상태 + target_id 결합
    states_tensor = torch.tensor(state).unsqueeze(0)  # [1, 28]
    targets_tensor = torch.tensor(target_onehot).unsqueeze(0)  # [1, 8]


    with torch.no_grad():
        q_values = model(states_tensor, targets_tensor)  # [1, action_size]
        q_values = q_values.numpy().flatten()
        exp_q = np.exp(q_values - np.max(q_values))  # for numerical stability
        softmax_probs = exp_q / np.sum(exp_q)

    plt.bar(range(len(softmax_probs)), softmax_probs)
    plt.xlabel("Action")
    plt.ylabel("Softmax Probability")
    plt.title("Softmax Action Probability Distribution")




def visualize_hidden_activations(model, input_dim, num_samples=1000):
    state_dim = 28
    target_dim = 8

    # 상태 28차원 무작위 생성
    states = np.random.rand(num_samples, state_dim).astype(np.float32)

    # 무작위 target_id 부여 및 one-hot 인코딩
    target_ids = np.random.randint(0, target_dim, size=(num_samples,))
    target_onehots = np.eye(target_dim)[target_ids].astype(np.float32)

    # 전체 입력 결합 (28 + 8 = 36)
    states_tensor = torch.tensor(states)
    targets_tensor = torch.tensor(target_onehots)

    with torch.no_grad():
        _, h1, h2, h3 = model(states_tensor, targets_tensor, return_hidden=True)

    h1 = h1.numpy()
    h2 = h2.numpy()
    h3 = h3.numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 5, 1)
    sns.heatmap(h1, cmap="viridis")
    plt.title("Hidden Layer 1 Activations")

    plt.subplot(1, 5, 2)
    sns.heatmap(h2, cmap="magma")
    plt.title("Hidden Layer 2 Activations")
    
    plt.subplot(1, 5, 3)
    sns.heatmap(h3, cmap="inferno")
    plt.title("Hidden Layer 3 Activations")

    plt.subplot(1, 5, 4)
    visualize_action_softmax_distribution(model, input_dim)
    plt.title("All Hidden Layer Activations")

    plt.tight_layout()
    plt.show()


def visualize_q_surface_1dim(model, input_dim, ax, action_index=1, resolution=50, slices=None):
    state_dim = 28
    target_dim = 8

    if state_dim < 3:
        print("⚠️ Q surface는 최소 3차원 상태가 필요합니다.")
        return

    if slices is None:
        slices = [1.0]

    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)

    for idx, slice_val in enumerate(slices):
        Z = np.zeros_like(X)

        for i in range(resolution):
            for j in range(resolution):
                state = np.ones((1, state_dim), dtype=np.float32) * 0.5
                state[0, 0] = X[i, j]
                state[0, 1] = Y[i, j]
                state[0, 2] = slice_val

                # 고정된 target_id = 0 사용 (or random 가능)
                target_id = 0
                target_onehot = np.eye(target_dim, dtype=np.float32)[target_id].reshape(1, -1)

                with torch.no_grad():
                    q_values = model(
                        torch.tensor(state, dtype=torch.float32),
                        torch.tensor(target_onehot, dtype=torch.float32)
                    )
                    Z[i, j] = q_values[0, action_index].item()

        ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

    ax.set_title(f"Q-value Surface for Action {action_index} (slices on state[2])")
    ax.set_xlabel("State[0]")
    ax.set_ylabel("State[1]")
    ax.set_zlabel("Q-value")



 
import torch
import torch.nn as nn
import numpy as np

def visualize_loss_surface_avg(model, input_dim, output_dim, ax, loss_fn=None, resolution=50, batch_size=32):
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    # model을 evaluation 모드로 설정 (Dropout, BatchNorm 등 비활성화)
    model.eval()
    state_dim = 28
    target_dim = 8


    # 고정된 batch of inputs and targets
    x_batch = torch.rand((batch_size, input_dim))
    target_q_batch = torch.rand((x_batch.shape[0], output_dim))  # 즉 [32, 6]
    states = x_batch[:, :state_dim]
    targets = x_batch[:, state_dim:]
    # 초기 weight 값
    w1 = model.fc1.weight.data.clone()
    param1 = w1[0, 0].item()
    param2 = w1[0, 1].item()

    # 변화 범위
    delta = 1.0
    p1_range = np.linspace(param1 - delta, param1 + delta, resolution)
    p2_range = np.linspace(param2 - delta, param2 + delta, resolution)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    Z = np.zeros_like(P1)

    with torch.no_grad():
        # 다른 weight는 고정
        original_weights = model.fc1.weight.data.clone()

        for i in range(resolution):
            for j in range(resolution):
                # fc1의 (0,0), (0,1)만 수정
                model.fc1.weight.data[0, 0] = P1[i, j]
                model.fc1.weight.data[0, 1] = P2[i, j]

                # forward 및 loss 계산
                q = model(states, targets)
                loss = loss_fn(q, target_q_batch)
                Z[i, j] = loss.item()

        # weight 원복
        model.fc1.weight.data = original_weights

    # 시각화
    ax.plot_surface(P1, P2, Z, cmap='plasma')
    ax.set_title("Average Loss Surface (2D Weight Slice)")
    ax.set_xlabel("Weight[0,0]")
    ax.set_ylabel("Weight[0,1]")
    ax.set_zlabel("Loss")


 
def visualize_loss_surface_full(model, input_dim, output_dim, ax, loss_fn=None, resolution=30, delta=1.0):
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    state_dim = 28
    target_dim = 8
    batch_size=32
    x_batch = torch.rand((batch_size, input_dim))
    states = x_batch[:, :state_dim]
    targets = x_batch[:, state_dim:]
    # 기준 파라미터 (백업)
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    # 랜덤 입력/출력
    x = torch.rand((1, input_dim))
    target_q = torch.rand((x_batch.shape[0], output_dim))  # 즉 [32, 6]

    # 랜덤 방향 2개 생성 (정규화)
    d1 = torch.randn_like(original_params)
    d1 = d1 / d1.norm()

    d2 = torch.randn_like(original_params)
    d2 = d2 - torch.dot(d1, d2) * d1  # d1과 직교하게 투영
    d2 = d2 / d2.norm()

    # 격자 좌표
    a = np.linspace(-delta, delta, resolution)
    b = np.linspace(-delta, delta, resolution)
    A, B = np.meshgrid(a, b)
    Z = np.zeros_like(A)

    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                theta = original_params + a[i] * d1 + b[j] * d2
                torch.nn.utils.vector_to_parameters(theta, model.parameters())

                output = model(states, targets)
                loss = loss_fn(output, target_q)
                Z[j, i] = loss.item()  # matplotlib는 [row, col] = [y, x]

    # 원래 파라미터 복원
    torch.nn.utils.vector_to_parameters(original_params, model.parameters())

    # 시각화
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, Z, cmap='inferno')
    ax.set_title("Loss Surface Over Full Parameter Space (Random 2D Slice)")
    ax.set_xlabel("Direction 1 (α)")
    ax.set_ylabel("Direction 2 (β)")
    ax.set_zlabel("Loss")

def visualize_q_surface_3dim(model, input_dim, ax, action_index=3, resolution=50, slice_val=0.5):
    state_dim = 28
    target_dim = 8

    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            state = np.ones((1, state_dim), dtype=np.float32) * 0.5
            state[0, 0] = X[i, j]
            state[0, 1] = Y[i, j]
            state[0, 2] = slice_val

            # 고정된 target_id 사용 (예: 0)
            target_id = 0
            target_onehot = np.eye(target_dim, dtype=np.float32)[target_id].reshape(1, -1)

            with torch.no_grad():
                q_values = model(
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(target_onehot, dtype=torch.float32)
                )
                Z[i, j] = q_values[0, action_index].item()

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.set_title(f"Q-value Surface for Action {action_index} (state[2]={slice_val:.2f})")
    ax.set_xlabel("State[0]")
    ax.set_ylabel("State[1]")
    ax.set_zlabel("Q-value")


# 통합 시각화 함수
def visualize_q_values(model_path, input_dim, output_dim, target_count=8):
    model, _, _ = load_model(model_path, input_dim, output_dim)
   
    
    plt.subplot(2, 3, 2)  # 2x3 그리드에서 두 번째 위치
    print("Visualizing Q-values across states...")
    visualize_q_across_states(model, input_dim, output_dim,)

    plt.subplot(2, 3, 3)  # 2x3 그리드에서 세 번째 위치
    print("Visualizing Q heatmap (PCA)...")
    visualize_q_pca_tsne(model, input_dim, method='pca')
    
    plt.subplot(2, 3, 4)  # 2x3 그리드에서 네 번째 위치
    print("Visualizing policy map (PCA)...")
    visualize_policy_boundary(model, input_dim, method='pca')
    

    visualize_target_id_effect(model, base_state=np.zeros(input_dim-8), num_targets=8)
    print("Visualizing Target ID Effect...")

    
    plt.tight_layout()  # 그래프 간 간격 조정
    plt.show()  # 한 번에 모든 그래프를 화면에 띄움
    
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 3, 1)  # 2x3 그리드에서 두 번째 위치
    print("Visualizing argmax actions...")
    visualize_argmax_actions(model, input_dim)
        
    print("Visualizing Q-values by state dimensions...")
    visualize_q_by_state_dims(model, input_dim, output_dim, dims=[16], num_states=1000)
    
    plt.subplot(2, 3, 3)  # 2x3 그리드에서 두 번째 위치
    print("Visualizing hidden layers by t-SNE...")
    plot_hidden_tsne(model, input_dim)
    
    plt.subplot(2, 3, 4)  # 2x3 그리드에서 두 번째 위치
    print("Visualizing hidden layers by piecewise structure...")
    plot_piecewise_structure(model, input_dim)
        
    plt.subplot(2, 3, 5)  # 2x3 그리드에서 두 번째 위치
    print("Visualizing hidden layers by function slope...")
    plot_function_slope(model, input_dim)
    
    plt.subplot(2, 3, 6)  # 2x3 그리드에서 두 번째 위치
    print("Visualizing hidden layers by relu regions...")
    plot_relu_regions(model, input_dim)
    
    plt.tight_layout()  # 그래프 간 간격 조정
    plt.show()  # 한 번에 모든 그래프를 화면에 띄움
    
    # print("Visualizing Q-values and policy 3D surface...")
    # visualize_q_and_policy_3d(model, input_dim, output_dim)

    fig = plt.figure(figsize=(15, 12))    
    
    ax = fig.add_subplot(2, 2, 1,projection='3d')  # 2x2 그리드에서 두 번째 위치
    print("Visualizing Q-value 3D surface for Action 2...")
    visualize_q_surface_1dim(model, input_dim, action_index=2,ax = ax)  
    
    bx = fig.add_subplot(2, 2, 2,projection='3d')  # 2x2 그리드에서 두 번째 위치
    print("Visualizing Q-value 3D surface for Action 3...")
    visualize_q_surface_3dim(model, input_dim, action_index=3,ax = bx)
    
    cx = fig.add_subplot(2, 2, 3,projection='3d')  # 2x2 그리드에서 두 번째 위치
    print("Visualizing Loss Surface (W[0,0] vs W[0,1])...")
    visualize_loss_surface_avg(model, input_dim, output_dim,ax = cx)
    
    dx = fig.add_subplot(2, 2, 4, projection='3d')
    print("Visualizing Full Parameter Loss Surface...")
    visualize_loss_surface_full(model, input_dim, output_dim,ax = dx)  
     
    plt.tight_layout()  # 그래프 간 간격 조정
    plt.show()  # 한 번에 모든 그래프를 화면에 띄움
    
    # plt.subplot(2, 2, 1)  # 2x2 그리드에서 두 번째 위치
    print("Visualizing Hidden Activations...")
    visualize_hidden_activations(model, input_dim)
  

def extract_index(filename):
    match = re.search(r"dqn_model(\d+)\.pth", filename)
    return int(match.group(1)) if match else -1

# 실행 메인
def main():
    INPUT_DIM = 36  # 입력 차원 (예시)
    OUTPUT_DIM = 6  # 출력 차원 (예시)
    TARGET_COUNT = 8
   
    # argparse 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, help="Model index to visualize", required=False)
    args = parser.parse_args()

    # 현재 디렉토리 내 .pth 파일 필터링
    current_dir = os.path.dirname(__file__)
    all_files = [f for f in os.listdir(current_dir) if f.endswith(".pth")]
    jfiles = [f for f in os.listdir(current_dir) if f.endswith(".json")]
    Nfiles = [f for f in os.listdir(current_dir) if f.endswith(".npy")]

    # 숫자 기준 정렬
    files = sorted(all_files, key=extract_index)

    if not Nfiles:
        print("No .npy files found.")
    else:
        print("Available .npy files:", Nfiles)
        visualize_n_state_time(Nfiles)

    if not jfiles:
        print("No .json files found.") 
    else:
        print("Available .json files:", jfiles)
        plt.figure(figsize=(20, 12))
        plt.subplot(2, 3, 1)  # 2x2 그리드에서 첫 번째 위치
        visualize_q_over_time(jfiles)

    if not files:
        print("No .pth files found.")
        return
    if args.model is None:
        print("Available .pth files:")
        for i, f in enumerate(files):
            print(f"{i}: {f}")
        try:
            idx = len(files) - 1  # 마지막 파일 선택
            print(f"Selected last model.%d {files[idx]}")
        except ValueError:
            print("Invalid input.")
            return
    else:
        idx = args.model

    if 0 <= idx < len(files):
        visualize_q_values(os.path.join(current_dir, files[idx]), INPUT_DIM, OUTPUT_DIM, TARGET_COUNT)
    else:
        print("Invalid model index.")
        return

if __name__ == "__main__":
    main()
