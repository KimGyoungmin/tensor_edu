import numpy as np
# 평균 제곱근 오차를 파이썬으로 구현
# 임의로 정한 기울기 a와 절편 b의 값이 각각 3과 76이라고 할 때 오차 구하기 y = 3x + 76
# 오차 1, -5, 3, 3

# 평균 제곱 오차
error = [1, -5, 3, 3]
print(np.array(error))
# 배열의 길이
n = len(error)

# 평균 제곱들의 합의 평균
hap_error = sum(x**2 for x in error) / n

print(f"평균 제곱 오차 : {hap_error}")

# 평균 제곱 오차에서 -> 제곱근으로 변환

root_hap_error = np.sqrt(hap_error)

print(f"평균 제곱근 오차 : {root_hap_error}")
# Numpy를 사용하여 RSME 만드는방법
print(np.sqrt(np.mean(np.square(np.array(error)))))