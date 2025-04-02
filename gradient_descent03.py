import matplotlib.pyplot as plt
import numpy as np 

data = [[1, 0.2], [2, 0.3], [3, 0.5], [4, 0.6], [5, 0.9], [6, 0.95], [7, 1.1], [8, 1.5]]

# 각 주의 대한 x 값들
x = [i[0] for i in data]
# 각 주마다의 물고기의 크기 y 값들
y = [i[1] for i in data]

plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.show()

x_data = np.array(x)
y_data = np.array(y)

# Learning Rate 설정
lr = 0.01

# epoch 설정
epochs = 1001

# 초기값 a와 b 값 설정
a = 0.5
b = 0.5

# 에포크만큼 반복
for i in range(epochs):
    # y 예측값 도출
    y_pred = a * x_data + b
    # 예측값과 데이터값의 오차 계산
    error = y_pred - y_data
    
    # MSE 비용 함수에 대한 a의 미분식 계산(a의 기울기)
    a_diff = (2/len(x_data) * np.sum(error * x_data))
    # MSE 비용 함수에 대한 b의 미분식 계산(b의 기울기)
    b_diff = (2/len(x_data) * sum(error))
    
    # a, b에 대한 예측값 업데이트
    a = a - lr * a_diff
    b = b - lr * b_diff
    
    
    # 에포크가 100단위 일때마다 실행
    if i % 100 == 0:
        # 오차함수 계산
        error_rate = np.mean(error**2)
        print(f"epoch = {i}, 기울기 = {round(a_diff, 4)}, 절편 = {round(b_diff, 4)}, 오차 = {round(error_rate, 4)}")
    
# 앞에서 최적의 a,b를 구한 값을 그래프에 대입
y_pred = a * x_data + b

plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()