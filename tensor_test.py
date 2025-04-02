# import tensorflow as tf
# tensorflow 버전 확인
# print(tf.__version__)

# 딥러닝을 구동하는 데 필요한 케라스 함수
from keras import Sequential
from keras.src.layers import Dense, Input

# 필요한 라이브러리
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정
np.random.seed(3)
tf.random.set_seed(3)

# 준비된 수술 환자 데이터 불러오기
data_set = np.loadtxt("my_data/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 x와 y로 구분하여 저장
x = data_set[:, 0:17]
y = data_set[:, 17]

# 딥러닝 구조를 결정(모델을 설정하고 실행하는 부분)
model = Sequential()
model.add(Input(shape=(17,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(x, y)[1]))