import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
if platform.system() == 'Windows':# 윈도우인 경우
    font_path = "c:/Windows/Fonts/malgun.ttf" # 맑은 고딕 폰트 경로
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
elif platform.system() == 'Darwin': # macOS인 경우
    plt.rc('font', family='AppleGothic')
else: # Linux 등 다른 OS인 경우
    plt.rc('font', family='NanumGothic')
# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
# 1. 이차 함수와 그 미분 정의
def f(x):
    return x**2

def df(x):
    return 2*x
# 2. 초기값과 학습률 설정
x = 3.0   # 시작점
alpha = 0.9 # 최적 학습률
# alpha = 1.1  # 발산 학습률
# alpha = 0.99  # 미적합 학습률
num_iterations = 50 # 반복 횟수

# 결과 기록용 리스트
x_history = [x]
f_history = [f(x)]
print(f"초기값: x = {x:.4f}, f(x) = {f(x):.4f}")
# 3. 경사하강법 반복
for i in range(num_iterations):
    gradient = df(x)
    x = x - alpha * gradient  # x 업데이트
    x_history.append(x)
    f_history.append(f(x))
    
    print(f"Iteration {i+1}: x = {x:.4f}, gradient = {gradient:.4f}, f(x) = {f(x):.4f}")

# 4. 함수 그래프와 이동 경로 시각화
x_vals = np.linspace(-3, 3, 100)
y_vals = f(x_vals)
plt.figure(figsize=(7,10))
plt.plot(x_vals, y_vals, label="f(x) = x^2")
# 경사하강법으로 이동한 지점들 표시
plt.plot(x_history, f_history, '.-', label="경사하강 이동 경로")

plt.title("경사하강법으로 x^2 최소화하는 과정")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
