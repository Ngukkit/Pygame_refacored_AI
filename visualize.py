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

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
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

# 모델 로드 함수
def load_model(model_path, input_dim, output_dim):
    model = DQN(input_dim, output_dim)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
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
    num_states = 1000  # 시각화할 상태 수
    states = np.random.rand(num_states, input_dim).astype(np.float32)
    with torch.no_grad():
        q_values = model(torch.tensor(states)).numpy()

    # plt.figure(figsize=(8, 6))
    for i in range(output_dim):
        plt.plot(states[:, 0], q_values[:, i], label=f'Action {i}')
    plt.title("Q-values across state space")
    plt.xlabel("State (normalized)")
    plt.ylabel("Q-value")
    plt.legend()
    plt.grid(True)


def visualize_q_and_policy_3d(model, input_dim, output_dim, method='tsne', num_states=1000):
    # 무작위 상태 생성
    states = np.random.rand(num_states, input_dim).astype(np.float32)
    with torch.no_grad():
        q_values = model(torch.tensor(states)).numpy()
        max_q = np.max(q_values, axis=1)
        actions = np.argmax(q_values, axis=1)

    # 차원 축소 (PCA 또는 t-SNE)
    if method == 'pca':
        reducer = PCA(n_components=3)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=30, n_iter=1000)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    reduced = reducer.fit_transform(states)

    # 시각화 시작
    fig = plt.figure(figsize=(14, 6))

    # 3D Q값 히트맵
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    p1 = ax1.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=max_q, cmap='viridis', s=20)
    ax1.set_title(f"Q-value Heatmap (3D {method.upper()})")
    ax1.set_xlabel("Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.set_zlabel("Dim 3")
    fig.colorbar(p1, ax=ax1, shrink=0.6, label="Max Q-value")

    # 3D 행동 결정 경계 (argmax action)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    cmap = plt.cm.get_cmap('tab10', output_dim)
    p2 = ax2.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=actions, cmap=cmap, s=20)
    ax2.set_title(f"Policy Map (3D {method.upper()})")
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.set_zlabel("Dim 3")
    fig.colorbar(p2, ax=ax2, shrink=0.6, label="Argmax Action")

    plt.tight_layout()
    plt.show()

    
def visualize_q_pca_tsne(model, input_dim, method='pca', num_states=1000):
    states = np.random.rand(num_states, input_dim).astype(np.float32)
    with torch.no_grad():
        q_values = model(torch.tensor(states)).numpy()
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
    states = np.random.rand(num_states, input_dim).astype(np.float32)
    with torch.no_grad():
        q_values = model(torch.tensor(states)).numpy()
        actions = np.argmax(q_values, axis=1)

    reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, perplexity=30)
    reduced = reducer.fit_transform(states)

    # plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=actions, cmap='tab10', s=15)
    plt.colorbar(label="Argmax Action")
    plt.title(f"Argmax Policy Map ({method.upper()})")
    # plt.legend()
    plt.grid(True)


def visualize_q_by_state_dims(model, input_dim, output_dim, dims=[0, 1, 2], num_states=1000):
    states = np.random.rand(num_states, input_dim).astype(np.float32)
    with torch.no_grad():
        q_values = model(torch.tensor(states)).numpy()
    aa = 1
    for d in dims:
        aa += 1
        plt.subplot(2, 2, aa)  # 2x2 그리드에서 두 번째 위치
        for i in range(output_dim):
            plt.scatter(states[:, d], q_values[:, i], label=f'Action {i}', s=10, alpha=0.5)
        plt.title(f"Q-values vs State Dimension {d}")
        plt.xlabel(f"State Dimension {d}")
        plt.ylabel("Q-value")
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
    num_states = 1000  # 시각화할 상태 수
    states = np.random.rand(num_states, input_dim).astype(np.float32)
    with torch.no_grad():
        q_values = model(torch.tensor(states))
        actions = torch.argmax(q_values, dim=1).numpy()

    # plt.figure(figsize=(8, 6))
    unique_actions = np.unique(actions)
    cmap = plt.cm.get_cmap('tab10', len(unique_actions))

    for action_id in unique_actions:
        idx = actions == action_id
        plt.scatter(
            states[idx, 0],
            [action_id] * np.sum(idx),
            c=np.full(np.sum(idx), action_id),  # 각 점에 action_id 값을 부여
            cmap=cmap,
            label=f"Action {action_id}",
            edgecolors='k',
            s=40
        )

    # plt.scatter(states[:, 0], actions, c=actions, cmap='tab10')
    plt.title("Argmax Actions by State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Argmax Action")
    plt.legend(title="Actions", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)


def visualize_hidden_activations(model, input_dim, num_samples=1000):
    states = np.random.rand(num_samples, input_dim).astype(np.float32)
    with torch.no_grad():
        _, h1, h2, h3, h4, h5 = model(torch.tensor(states), return_hidden=True)

    h1 = h1.numpy()
    h2 = h2.numpy()
    h3 = h3.numpy()
    h4 = h4.numpy()
    h5 = h5.numpy()

    plt.figure(figsize=(12, 5))

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
    sns.heatmap(h4, cmap="plasma")
    plt.title("Hidden Layer 4 Activations")

    plt.subplot(1, 5, 5)
    sns.heatmap(h5, cmap="cividis")
    plt.title("Hidden Layer 5 Activations")

    plt.tight_layout()
    plt.show()


def visualize_q_surface(model, input_dim, ax,action_index=0, resolution=50):
    if input_dim < 2:
        print("⚠️ Q surface는 최소 2차원 입력 상태가 필요합니다.")
        return

    # x축, y축용 2D 상태 공간 샘플 생성 (앞의 두 차원만 사용)
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 전체 상태는 input_dim 차원. 앞의 두 개만 다르게 만들고 나머지는 0.5로 고정.
    for i in range(resolution):
        for j in range(resolution):
            state = np.ones((1, input_dim), dtype=np.float32) * 0.5
            state[0, 0] = X[i, j]
            state[0, 1] = Y[i, j]
            with torch.no_grad():
                q_values = model(torch.tensor(state))
                Z[i, j] = q_values[0, action_index].item()

    # 3D surface plot
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(f"Q-value Surface for Action {action_index}")
    ax.set_xlabel("State[0]")
    ax.set_ylabel("State[1]")
    ax.set_zlabel("Q-value")

 
def visualize_loss_surface(model, input_dim, output_dim, ax, loss_fn=None, resolution=50):
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    # 가짜 입력/정답 생성
    x = torch.rand((1, input_dim))
    target_q = torch.rand((1, output_dim))

    # 분석할 weight parameter 2개 선택 (예: fc1 첫 weight 2개)
    w1 = model.fc1.weight.data.clone()
    param1 = w1[0, 0].item()
    param2 = w1[0, 1].item()

    delta = 1.0  # 변화 범위
    p1_range = np.linspace(param1 - delta, param1 + delta, resolution)
    p2_range = np.linspace(param2 - delta, param2 + delta, resolution)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    Z = np.zeros_like(P1)

    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                # weight 변경
                model.fc1.weight[0, 0] = P1[i, j]
                model.fc1.weight[0, 1] = P2[i, j]

                # 정방향 계산 및 손실
                q = model(x)
                loss = loss_fn(q, target_q)
                Z[i, j] = loss.item()

    # 시각화
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P1, P2, Z, cmap='plasma')
    ax.set_title("Loss Surface (2D Weight Slice)")
    ax.set_xlabel("Weight[0,0]")
    ax.set_ylabel("Weight[0,1]")
    ax.set_zlabel("Loss")

 
def visualize_loss_surface_full(model, input_dim, output_dim, ax, loss_fn=None, resolution=30, delta=1.0):
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    # 기준 파라미터 (백업)
    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    # 랜덤 입력/출력
    x = torch.rand((1, input_dim))
    target_q = torch.rand((1, output_dim))

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

                output = model(x)
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

def visualize_loss_surface_full_with_contour(model, input_dim, output_dim, ax, loss_fn=None, resolution=30, delta=1.0):
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    original_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    x = torch.rand((1, input_dim))
    target_q = torch.rand((1, output_dim))

    d1 = torch.randn_like(original_params)
    d1 = d1 / d1.norm()
    d2 = torch.randn_like(original_params)
    d2 = d2 - torch.dot(d1, d2) * d1
    d2 = d2 / d2.norm()

    a = np.linspace(-delta, delta, resolution)
    b = np.linspace(-delta, delta, resolution)
    A, B = np.meshgrid(a, b)
    Z = np.zeros_like(A)

    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                theta = original_params + a[i] * d1 + b[j] * d2
                torch.nn.utils.vector_to_parameters(theta, model.parameters())
                output = model(x)
                loss = loss_fn(output, target_q)
                Z[j, i] = loss.item()

    torch.nn.utils.vector_to_parameters(original_params, model.parameters())

    # 곡면 + 등고선 함께 시각화
    surf = ax.plot_surface(A, B, Z, cmap='viridis', alpha=0.9)
    ax.contour(A, B, Z, zdir='z', offset=Z.min() - 0.1, cmap='coolwarm')  # 등고선 추가
    ax.set_title("Loss Surface with Contour")
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_zlabel("Loss")
    ax.set_zlim(Z.min() - 0.1, Z.max())


   
# 통합 시각화 함수
def visualize_q_values(model_path, input_dim, output_dim):
    model, _, _ = load_model(model_path, input_dim, output_dim)
   
    
    plt.subplot(2, 2, 2)  # 2x2 그리드에서 두 번째 위치
    print("Visualizing Q-values across states...")
    visualize_q_across_states(model, input_dim, output_dim)

    plt.subplot(2, 2, 3)  # 2x2 그리드에서 세 번째 위치
    print("Visualizing Q heatmap (PCA)...")
    visualize_q_pca_tsne(model, input_dim, method='pca')
    
    plt.subplot(2, 2, 4)  # 2x2 그리드에서 네 번째 위치
    print("Visualizing policy map (PCA)...")
    visualize_policy_boundary(model, input_dim, method='pca')
    
    plt.tight_layout()  # 그래프 간 간격 조정
    plt.show()  # 한 번에 모든 그래프를 화면에 띄움
    
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 2, 1)  # 2x2 그리드에서 두 번째 위치
    print("Visualizing argmax actions...")
    visualize_argmax_actions(model, input_dim)
        
    print("Visualizing Q-values by state dimensions...")
    visualize_q_by_state_dims(model, input_dim, output_dim, dims=[0, 1, 2])
    
    plt.tight_layout()  # 그래프 간 간격 조정
    plt.show()  # 한 번에 모든 그래프를 화면에 띄움
    
    print("Visualizing Q-values and policy 3D surface...")
    visualize_q_and_policy_3d(model, input_dim, output_dim)

    fig = plt.figure(figsize=(15, 12))    
    
    ax = fig.add_subplot(2, 2, 1,projection='3d')  # 2x2 그리드에서 두 번째 위치
    print("Visualizing Q-value 3D surface for Action 0...")
    visualize_q_surface(model, input_dim, action_index=0,ax = ax)  
    
    bx = fig.add_subplot(2, 2, 2,projection='3d')  # 2x2 그리드에서 두 번째 위치
    print("Visualizing Loss Surface (W[0,0] vs W[0,1])...")
    visualize_loss_surface(model, input_dim, output_dim,ax = bx)
    
    cx = fig.add_subplot(2, 2, 3,projection='3d')  # 2x2 그리드에서 두 번째 위치
    print("Visualizing Full Parameter Loss Surface...")
    visualize_loss_surface_full(model, input_dim, output_dim,ax = cx)
    
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    print("Visualizing Loss Surface Random ...")
    visualize_loss_surface_full_with_contour(model, input_dim, output_dim, ax=ax)
    
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
    INPUT_DIM = 91  # 입력 차원 (예시)
    OUTPUT_DIM = 6  # 출력 차원 (예시)
   
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



    if not jfiles:
        print("No .json files found.") 
    else:
        print("Available .json files:", jfiles)
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 2, 1)  # 2x2 그리드에서 첫 번째 위치
        visualize_q_over_time(jfiles)
    if not Nfiles:
        print("No .npy files found.")
    else:
        print("Available .npy files:", Nfiles)
        # visualize_n_state_time(Nfiles)

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
        visualize_q_values(os.path.join(current_dir, files[idx]), INPUT_DIM, OUTPUT_DIM)
    else:
        print("Invalid model index.")
        return

if __name__ == "__main__":
    main()
