import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) # 공부하는 시간
y = np.array([11, 22, 33, 44, 53, 66, 77, 87, 95]) # 각 공부하는 시간에 맵핑되는 성적

inputs = Input(shape=(1,))
output = Dense(1, activation='linear')(inputs)
linear_model = Model(inputs, output)

sgd = optimizers.SGD(learning_rate=0.01)

linear_model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
linear_model.fit(x, y, epochs=300, verbose=2)