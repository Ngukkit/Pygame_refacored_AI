import re
import matplotlib.pyplot as plt

# 로그 파일에서 데이터 읽기
with open('game_log.txt', 'r', encoding='utf-8') as file:
    log_data = file.read()

# selected 값 추출
selected_values = list(map(int, re.findall(r'selected:(\d)', log_data)))

# 기대 확률 (0~4)
expected_probs = [0.4, 0.3, 0.1, 0.1, 0.1]

# 각 숫자의 누적 선택 수 계산
counts = [0] * 5
ratios_over_time = [[] for _ in range(5)]

# 누적 비율 계산
for i, val in enumerate(selected_values):
    counts[val] += 1
    total = i + 1
    for j in range(5):
        ratios_over_time[j].append(counts[j] / total)

# 그래프 출력
x = list(range(1, len(selected_values) + 1))
for i in range(5):
    plt.plot(x, ratios_over_time[i], label=f'Item {i} (Expected: {expected_probs[i]*100:.0f}%)')

# 기대값에 해당하는 수평선 그리기
for prob in expected_probs:
    plt.axhline(y=prob, color='gray', linestyle='--', linewidth=0.5)

plt.xlabel('Trial Count')
plt.ylabel('Selection Ratio')
plt.title('Convergence to Expected Probabilities (Law of Large Numbers)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
