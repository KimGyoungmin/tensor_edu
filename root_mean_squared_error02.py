import numpy as np

# 기울기 a와 y 절편 b
ab = [3, 76]

# x, y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y_true = [i[1] for i in data]

# y = ax + b에 a와 b 값을 대입하여 결과를 출력하는 함수

def predict(x):
    y_pred = [ab[0] * i + ab[1] for i in x]
    return y_pred

# 평균 제곱 오차
np_mse = np.mean(np.square(np.array(y_true) - np.array(predict(x))))
# 평균 제곱근 오차
np_rmse = np.sqrt(np_mse)

print(f"평균 제곱 오차 : {np_mse} \n 평균 제곱근 오차 : {np_rmse}")



