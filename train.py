# from agent_dualreplay import DualReplayAgent
from agent_dualreplay_lstm import DualReplayAgentLSTM
# from agent_prioritizedreplay import DQNAgent
# from agent import DQNAgent
from collections import deque
from game_env import MyGameEnv
from save_utils import initialize_training, log_q_values
import torch
import random
import numpy as np
import os
import re

# 게임 환경과 에이전트 연결
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
env = MyGameEnv(render_mode=True) 
 # 상태 벡터 크기
state,target_onehot = env.observe_state()
state_size = len(state)

action_size = 6  # 예시: 5가지 행동 (이동, 공격 등)

goal_indices = [
    2,    # 체력
    6, 7, # v_threat, h_threat (위협 상태 변화 감지)
    8,    # 공격력 변화 (raw_value)
    9, 11, 13, 15,  # 플랫폼 거리
    18, 20,         # 몬스터 1,2 거리
    19, 21,         # 몬스터 1,2 체력
    22, 23, 24, 25, 26,  # 아이템 거리
    27   # 포탈 거리
]


# 사용 시 agent = DQNAgent(..., goal_indices=goal_indices)


action_descriptions = {
    0: "<---",   # 예: '0'이면 '공격'을 수행
    1: "--->",   # 예: '1'이면 '점프'를 수행
    2: "Jump",  # 예: '2'이면 '스킬1'을 수행
    3: "attack",  # 예: '3'이면 '스킬2'를 수행
    4: "idle",  # 예: '4'이면 '스킬3'을 수행
    5: "use skill1",  # 예: '5'이면 '스킬4'를 수행
}

# trajectory 버퍼
traj_buffer = {k: [] for k in ['states', 'target_onehots', 'actions', 'rewards', 'next_states', 'dones', 'target_ids']}

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA 사용 가능?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("사용 중인 GPU:", torch.cuda.get_device_name(0))

agent = DualReplayAgentLSTM(state_size, action_size, goal_indices, target_count=8, device=device,lr =0.0001, traj_len=30)

# 모델을 디바이스로 이동
agent.model.to(device)

model_dir = "."

model_files = [f for f in os.listdir(model_dir) if re.match(r"dqn_model(\d+).pth", f)]

if model_files:
    # 모델 파일 이름에서 숫자 부분을 추출하여 정수로 변환 후, 가장 큰 숫자를 찾음
    epoch_numbers = [int(re.search(r"(\d+)", f).group(1)) for f in model_files]
    latest_epoch = max(epoch_numbers)
    
    # 가장 높은 숫자의 모델 파일 경로
    latest_model_path = f"dqn_model{latest_epoch}.pth"

    # 모델 불러오기
    agent.load(latest_model_path)
    print(f"모델 {latest_epoch}을(를) 불러왔습니다.")
else:
    print("모델 파일이 없어 새로 학습을 시작합니다.")
   
# 에피소드 수
episodes = 10000
# LSTM 입력 시퀀스를 위한 버퍼 (초기화)
state_seq_buffer = deque(maxlen=agent.traj_len)
target_seq_buffer = deque(maxlen=agent.traj_len)
start_epoch = latest_epoch if model_files else 0

# 상태를 Tensor로 변환하고 디바이스에 맞춰 이동
state_tensor = torch.tensor(state).unsqueeze(0).float().to(device)

for e in range(start_epoch,start_epoch + episodes):

    state,target_onehot = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    reward = 0 
    last_reward_step = 0  # 마지막 보상 스텝 초기화
        
    while not done:
    # 다음 액션을 위해 마지막 액션 시간 갱신
        state_seq_buffer.append(state)
        target_seq_buffer.append(target_onehot)
        
        # 시퀀스가 최소 길이 이상일 때만 행동 선택
        if len(state_seq_buffer) >= agent.traj_len:
            # numpy or torch 텐서로 변환
            state_seq_np = np.array(state_seq_buffer)  # 리스트 -> ndarray 변환
            target_seq_np = np.array(target_seq_buffer)  # 리스트 -> ndarray 변환
            state_seq_tensor = torch.tensor(state_seq_np, dtype=torch.float32).unsqueeze(0).to(device)
            target_seq_tensor = torch.tensor(target_seq_np, dtype=torch.float32).unsqueeze(0).to(device)

            action = agent.act(state_seq_tensor, target_seq_tensor, epsilon=0.1)
        else:
            # 랜덤 액션 또는 idle
            action = random.randint(0, action_size - 1)
        # action = agent.act(state, target_onehot, epsilon=0.1)
        action_description = action_descriptions.get(action, "알 수 없는 액션")

        next_state,target_onehot, reward, done, _ = env.step(action)
        # latest_epoch = max(epoch_numbers) if model_files else 0

        step_count +=1

        if (step_count - last_reward_step) % 500 == 0 and step_count != last_reward_step:
            print(f"total steps: {step_count - last_reward_step} 스텝 epsilon UP ")
            # agent.epsilon = min(agent.epsilon + 0.005, 1)
            # print(f"postion : x{env.px} ,y{env.py}\n")
            reward = -0.5
       
        if step_count - last_reward_step == 1500:
            print(f"보상 없이 1,500 스텝 초과 강제 종료, x{env.px} ,y{env.py}")
            reward = -1.0
            done = True
  
        # if step_count - last_reward_step >= 3000:
        #     print(f"보상 없이 3,000 스텝 초과, 에피소드 강제 종료 x{env.px} ,y{env.py}")
        #     reward = -1.0
        #     done = True
        
        # if reward >= 0.05:
        #     last_reward_step = step_count
            # agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.999) # 보상 받으면 탐험 감소
        if reward > 0:
            # print(f" reward: {reward}  x{env.px} ,y{env.py}")
            # agent.update_schedules()
            last_reward_step = step_count
            
        
        env.last_reward_timer(step_count-last_reward_step,MAXCOUNTDOWN=1500)  # 보상 타이머 설정

        # step 루프 내부에서:
        traj_buffer['states'].append(state)
        traj_buffer['target_onehots'].append(target_onehot)
        traj_buffer['actions'].append(action)
        traj_buffer['rewards'].append(reward)
        traj_buffer['next_states'].append(next_state)
        traj_buffer['dones'].append(done)
        traj_buffer['target_ids'].append(env.target_id)
     
        # 일정 길이 도달 시 trajectory 저장
        if len(traj_buffer['states']) >= agent.traj_len:
            trajectory = {k: v[-agent.traj_len:] for k, v in traj_buffer.items()}
            trajectory['state_seq'] = list(state_seq_buffer)[-agent.traj_len:]
            trajectory['target_seq'] = list(target_seq_buffer)[-agent.traj_len:]
            agent.remember_trajectory(trajectory)

        # 기존 traj_buffer 저장 아래에 추가
        goal_delta = np.array(next_state)[goal_indices] - np.array(state)[goal_indices]
        agent.reward_buffer.add((state, action, goal_delta, env.target_id))

        if len(agent.reward_buffer) >= agent.batch_size:
            agent.train_reward_net()

        # 🔹 경험 저장     
        # agent.remember(state, action, reward, next_state, done, env.target_id) 
        
        # 경험 리플레이
        if  step_count % 100 == 0:
            # agent.replay()  
            agent.replay_from_trajectory()
        
        # 🔹 타겟 모델 업데이트
        if step_count % 300 == 0:
            agent.update_target_network()
        
        # 🔹 스케줄 갱신
        
        if total_reward < 0:
            print("reward is Zero, episode end")
            done = True
            
        if step_count % 100 == 0:
            # print(f"epsilon:{agent.epsilon:.4f},q_values:{agent.q_valuefordp:.4f},,x{env.px:3.0f},y{env.py:3.0f},tau:{agent.tau:.2f},v_threat:{env.v_threat},Reward:{total_reward:.3f},Action:{action_description}")
            print(f"q_values:{agent.q_values_view.max():.4f},x{env.px:3.0f},y{env.py:3.0f},v_threat:{env.v_threat},Reward:{total_reward:.3f},Action:{action_description}")
        # 죽었을때 모델 저장
        if done:
            # agent.epsilon = min(agent.epsilon_max, agent.epsilon * 1.01)  # 또는 1.05 증가도 가능
            # agent.epsilon = min(agent.epsilon_max, 0.5) # 에피소드 종료 시 탐험 증가
            log_q_values(agent, env, state, step_count, device=device)
            
        total_reward += reward
                
        state = next_state        

    # if (e + 1) % 20 == 0 and agent.epsilon > 0.8:
    #     save_path = rf"dqn_model{e+1}.pth"
    #     agent.save(save_path)
    #     print(f"{save_path} 저장 완료.")
    # elif (e + 1) % 50 == 0 and 0.8 >= agent.epsilon > 0.5:
    #     save_path = rf"dqn_model{e+1}.pth"
    #     agent.save(save_path)
    #     print(f"{save_path} 저장 완료.")
    # elif (e + 1) % 100 == 0 and 0.5 >= agent.epsilon:
    if (e + 1) % 100 == 0:
        initialize_training(e, agent, log_filename="q_values_log.json", device=device)

    #     model_filename = f"dqn_model{e+1}.pth"
    #     agent.save(model_filename)
    #     print(f"{model_filename} 저장 완료.")

    print(f"Episode {e+1}, End Total Reward: {total_reward:.3f}")

agent.save(r"dqn_model.pth")
