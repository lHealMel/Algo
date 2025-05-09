from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential, load_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input


mnist = fetch_openml('mnist_784')

train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=0.3, random_state=10, shuffle=True)

scaler = StandardScaler()
# Fit on training set.
scaler.fit(train_img)
# Apply transform to both the training and test sets
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# Make an instance of the Model
pca = PCA(.95)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# name refatcoring
x_train = train_img
y_train = to_categorical(train_lbl)
x_test = test_img
y_test = to_categorical(test_lbl)



print(x_train.shape)

model_weight_path = "data/mnist_pca_sgd.weights.h5"
model_path = "data/mnist_pca_sgd.keras"

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

#multivariate regression

#sgd = optimizers.SGD(learning_rate=0.01)
# model = load_model(model_path)
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_data=(x_test, y_test), callbacks=[early_stopping, checkpoint])


model.save(model_path)