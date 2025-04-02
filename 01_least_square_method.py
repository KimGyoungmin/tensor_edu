import numpy as np

# 회귀 분석 y= ax + b -> y(종속 변수)=성적 x(독립 변수)=공부한시간

# 공부한시간 2, 4, 6, 8
x = [2, 4, 6, 8]
# 성적 81, 93, 91, 97
y = [81, 93, 91, 97]

# x,y에 대한 평균 값 계산
# avg_x = sum([time for time in x]) / len(x)
# avg_y = sum([grade for grade in y]) / len(y)

# for i in range(len(x)):
#     # x와 y에 대한 합계
#     avg_x, avg_y = avg_x + x[i], avg_y + y[i]
# avg_x, avg_y = avg_x / len(x), avg_y / len(y)

# numpy의 메서드를 통해 평균을 쉽게 도출 가능
mx = np.mean(x)
my = np.mean(y)


# 최소 제곱법의 분모 구하기
deno = sum([(i-mx)**2 for i in x])

# 최소 제곱법의 분자 구하기
hap_x = [i-mx for i in x]
hap_y = [i-my for i in y]
# mole = 0
# for i in range(len(x)):
#     dataset = hap_x[i] * hap_y[i]
#     mole += dataset
# 리스트 컴프리헨션을 사용한 방법
mole = sum([x*y for x, y in zip(hap_x,hap_y)])

# 기울기 a의 값 구하기
a = mole / deno

# 절편 b의 값 구하기
b = my - mx*a

print(f"최소 제곱법을 이용한 회귀분석 : y = {a}x + {b}")