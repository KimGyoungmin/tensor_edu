import numpy as np

# 사육기간
x = [1, 2, 3, 4, 5, 6, 7, 8]
# 물고기의 크기
y = [0.2, 0.3, 0.5, 0.6, 0.9, 0.95, 1.1, 1.5]
# 알고자하는 주의 물고기 크기
input_month = int(input("week value : "))

# 평균 사육기간
mx = np.mean(x)
# 평균 물고기의 크기
my = np.mean(y)


def calculate_regression_params(x, y, mx, my):
    # 기울기 분모 계산
    demo = sum([(i-mx)**2 for i in x])
    # 기울기 분자 계산
    # x와 y에 대한 평균값을 뺀 값들의 리스트
    hap_x = [i-mx for i in x]
    hap_y = [i-my for i in y]

    # 각각의 도출된 x,y 곱을 더한다
    mole = sum(x*y for x,y in zip(hap_x, hap_y))
    # 기울기 a의 값 도출
    a = mole / demo
    # 절편 b의 값 도출
    b = my - mx*a
    return a, b

def predict_size(input_month):    
    a, b = calculate_regression_params(x, y, mx, my)
    size = a * input_month + b
    if size >= 30:
        size = 30
    return round(size, 3)


print(f"물고기의 {input_month}주 후의 예상 크기는 {predict_size(input_month)}입니다\n 1. 15, 22, 77, 200주 후의 크기 예상 : {predict_size(15)},  {predict_size(22)},  {predict_size(77)},  {predict_size(200)}")
print(f"5주차 예상 크기와 실제 크기의 차이 : {round(abs(predict_size(5) - y[4]), 3)}")
