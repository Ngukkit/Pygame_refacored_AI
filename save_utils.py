import os
import json
import torch

import os
import json

def initialize_training(e, agent, log_filename="q_values_log.json", device="cpu"):
    model_filename = f"dqn_model{e+1}.pth"
    before_filename = f"dqn_model{(e+1)-100}.pth"
    current_dir = os.path.dirname(__file__)
    all_files = [f for f in os.listdir(current_dir) if f.endswith(".pth")]

    # 모델 파일이 하나도 없으면 초기 학습 상태로 판단
    if len(all_files) == 0:
        print(f"{model_filename} 파일이 없어 초기 학습 상태입니다.")
        
        # 로그 파일 초기화 (있으면 삭제)
        if os.path.exists(log_filename):
            try:
                os.remove(log_filename)
                print(f"{log_filename} 파일 삭제 완료 (초기화).")
            except Exception as ex:
                print(f"{log_filename} 삭제 실패: {ex}")
        else:
            print(f"{log_filename} 파일이 없어 새로 생성합니다.")
        
        # 빈 로그 파일 새로 생성
        with open(log_filename, "w") as f:
            pass  # 빈 파일 생성

    # 현재 모델 저장
    agent.save(model_filename)
    print(f"{model_filename} 저장 완료.")
    
    # 이전 모델 삭제 (100 에피소드 전 것)
    if os.path.exists(before_filename):
        try:
            os.remove(before_filename)
            print(f"{before_filename} 파일 삭제 완료.")
        except Exception as ex:
            print(f"{before_filename} 삭제 실패: {ex}")



def log_q_values(agent, env, state, step_count, device="cpu", log_filename="q_values_log.json"):
    target_onehot = torch.nn.functional.one_hot(
        torch.tensor([env.target_id], dtype=torch.long), 
        num_classes=agent.target_count
    ).float().to(device)
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    q_values, _ = agent.model(state_tensor, target_onehot)
    q_values = q_values.detach().cpu().tolist()
    
    q_log = {
        "step": step_count,
        "q_values": q_values
    }
    
    with open(log_filename, "a") as f:
        json.dump(q_log, f)
        f.write("\n")