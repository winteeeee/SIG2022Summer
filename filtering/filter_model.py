from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import os
import pandas as pd
import Word2VecEncoder
import numpy as np

# Word2Vec 인코딩 + 케라스 모델
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("데이터 로드...\n")
data = pd.read_csv('C:\\Users\\Han SeongMin\\IdeaProjects\\SIG2022Summer\\filtering\\Unsmile_Data_Set_edudat.csv',
                   encoding='utf-8')
data.dropna()
xdata = data['sentences'].values
xdata = Word2VecEncoder.get_sentence_list(xdata)
xdata = Word2VecEncoder.get_embedded_sentence(xdata)
ydata = data['clean'].values
print("데이터 로드 완료\n")

print("학습, 훈련 데이터 나누는 중...\n")
xTrain, xTest, yTrain, yTest = train_test_split(xdata, ydata, test_size=0.2)
xTrain = np.asarray(xTrain, dtype=object).astype(float)
xTest = np.asarray(xTest, dtype=object).astype(float)
yTrain = np.asarray(yTrain).astype(float)
yTest = np.asarray(yTest).astype(float)
print("학습, 훈련 데이터 나누기 완료\n")

print("모델 생성...\n")
hidden_units = 32
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=100))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print("모델 생성 완료\n")

model.summary()
print("모델 함수 설정 중\n")
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=[metrics.binary_accuracy])
print("함수 설정 완료. 모델 학습 시작\n")
model.fit(xTrain, yTrain, batch_size=32, epochs=100, verbose=2, class_weight={0: 0.25, 1: 0.75})
model.evaluate(xTrain, yTrain)
model.evaluate(xTest, yTest)

print(xTrain[0])
print(yTest[:20])
print(model.predict(xTest[:20]))
model.save("filter_model3.h5")
