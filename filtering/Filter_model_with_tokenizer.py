import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
import os

# 정수 인코딩 + RNN 모델 사용되지 않음.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("데이터 로드...\n")
data = pd.read_csv('C:\\Users\\Han SeongMin\\IdeaProjects\\SIG2022Summer\\filtering\\Unsmile_Data_Set_edudat.csv',
                   encoding='utf-8')
data.dropna()
xdata = data['sentences']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(xdata)
xTrain = tokenizer.texts_to_sequences(xdata)
max_len = 41
xTrain = pad_sequences(xTrain, max_len)

ydata = data['clean']
print("데이터 로드 완료\n")

print("학습, 훈련 데이터 나누는 중...\n")
xTrain, xTest, yTrain, yTest = train_test_split(xTrain, ydata, test_size=0.2)
print("학습, 훈련 데이터 나누기 완료\n")

print("모델 생성...\n")
embedding_dim = 32
hidden_units = 32
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index) + 1
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))
print("모델 생성 완료\n")

model.summary()
print("모델 함수 설정 중\n")
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics='acc')
print("함수 설정 완료. 모델 학습 시작\n")
model.fit(xTrain, yTrain, batch_size=32, epochs=5, verbose=2, class_weight={0: 0.25, 1: 0.75})

model.evaluate(xTrain, yTrain)
model.evaluate(xTest, yTest)
#model.save("filter_model2.h5")
