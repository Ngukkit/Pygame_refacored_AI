from agent import DQNAgent
from game_env import MyGameEnv
import torch
# import numpy as np
import json
import os
import re

# 게임 환경과 에이전트 연결
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
env = MyGameEnv(render_mode=True)
state_size = len(env.observe_state())  # 상태 벡터 크기
state = env.observe_state()
action_size = 6  # 예시: 5가지 행동 (이동, 공격 등)

action_descriptions = {
    0: "<---",   # 예: '0'이면 '공격'을 수행
    1: "--->",   # 예: '1'이면 '점프'를 수행
    2: "Jump",  # 예: '2'이면 '스킬1'을 수행
    3: "attack",  # 예: '3'이면 '스킬2'를 수행
    4: "idle",  # 예: '4'이면 '스킬3'을 수행
    5: "use skill1",  # 예: '5'이면 '스킬4'를 수행
}

agent = DQNAgent(state_size, action_size)
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
episodes = 1000
start_epoch = latest_epoch if model_files else 0

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA 사용 가능?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("사용 중인 GPU:", torch.cuda.get_device_name(0))

# 모델을 디바이스로 이동
agent.model.to(device)

# 상태를 Tensor로 변환하고 디바이스에 맞춰 이동
state_tensor = torch.tensor(state).unsqueeze(0).float().to(device)

for e in range(start_epoch,start_epoch + episodes):

    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    reward = 0 
    last_reward = 0  # 마지막 보상 초기화
    last_reward_step = 0  # 마지막 보상 스텝 초기화
        
    while not done:
                
        action = agent.act(state,reward = reward,use_reward_simulation=False)
        action_description = action_descriptions.get(action, "알 수 없는 액션")

    # 다음 액션을 위해 마지막 액션 시간 갱신
        next_state, reward, done, _ = env.step(action)
        
        # latest_epoch = max(epoch_numbers) if model_files else 0

        step_count +=1

        if (step_count - last_reward_step) % 500 == 0 and step_count != last_reward_step:
            print(f"total steps: {step_count - last_reward_step} 스텝 epsilon UP ")
            # agent.epsilon = min(agent.epsilon + 0.005, 1)
            # print(f"postion : x{env.px} ,y{env.py}\n")
            reward = -1
       
        if step_count - last_reward_step == 1500:
            print(f"보상 없이 1,500 스텝 초과, x{env.px} ,y{env.py}")
            reward = -5
  
        if step_count - last_reward_step >= 3000:
            print(f"보상 없이 3,000 스텝 초과, 에피소드 강제 종료 x{env.px} ,y{env.py}")
            reward = -10
            done = True
            
        if reward >= 5:
            last_reward_step = step_count
            # agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.999) # 보상 받으면 탐험 감소
        elif reward > 0:
            # print(f" reward: {reward}  x{env.px} ,y{env.py}")
            last_reward_step = step_count
            
        state = next_state  # 상태 업데이트
        total_reward += reward
        last_reward = reward  # 마지막 보상 업데이트
        
        env.last_reward_timer(last_reward_step,MAXCOUNTDOWN=3000)  # 보상 타이머 설정

          
        agent.remember(state, action, reward, next_state, done)  # 경험 저장
  
        if len(agent.memory) >= 50 and step_count % 100 == 0:
            agent.replay(batch_size=32)  # 경험 리플레이

        if step_count % 300 == 0:
            agent.update_target_model()  # 타겟 모델 업데이트
        

        # if np.random.rand() < 0.01:
        #     agent.log_rewardnet_outputs("rewardnet_log.npy")
      
            
        if step_count % 50 == 0:
            print(f"epsilon:{agent.epsilon:.4f},q_values:{agent.q_valuefordp:.4f},x{env.px:3.0f},y{env.py:3.0f},Reward:{total_reward:2d},Action:{action_description}")

        if done:
            # agent.epsilon = min(agent.epsilon_max, agent.epsilon * 1.01)  # 또는 1.05 증가도 가능
            # agent.epsilon = min(agent.epsilon_max, 0.5) # 에피소드 종료 시 탐험 증가
        
            q_log = {
                "step": step_count,
                "q_values": agent.model(state_tensor).detach().cpu().tolist()
                }  # Q 값 로그      
            with open("q_values_log.json", "a") as f:
                json.dump(q_log, f)
                f.write("\n")

    if (e + 1) % 10 == 0:
        save_path = rf"dqn_model{e+1}.pth"
        agent.save(save_path)
        print(f"{save_path} 저장 완료.")

    print(f"Episode {e+1}, Total Reward: {total_reward}")

agent.save(r"dqn_model.pth")
