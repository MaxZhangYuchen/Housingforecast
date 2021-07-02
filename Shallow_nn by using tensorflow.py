import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense

# 数据处理
data = pd.read_csv('bj_housing.csv')
Y_true = data[['Value']]
X = data.drop('Value', axis=1)
minimum_price = np.min(Y_true)
maximum_price = np.max(Y_true)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
Y_true = scaler.fit_transform(Y_true)

# 数据集乱序
np.random.seed(116)
np.random.shuffle(X)
np.random.seed(116)
np.random.shuffle(Y_true)
tf.random.set_seed(116)

# 规范x和y的数据类型
X = tf.cast(X, tf.float32)
Y_true = tf.cast(Y_true, tf.float32)

# 生成Sequential 顺序模型
model = tf.keras.Sequential()

# create two layer
layer1 = Dense(120, input_shape=(6,), activation='relu')   # Hidden layer
layer2 = Dense(1, activation='relu')                       # Output layer

model.add(layer1)
model.add(layer2)


model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=0.01),
              loss='mse')

# 切出20%数据作为训练集，每20轮训练一次，总共训练200轮，每组11个数据
history = model.fit(X, Y_true, validation_split=0.20, epochs=200, batch_size=10, validation_freq=1)

model.summary()   # 生成报告

# 画出损失函数图像
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(np.arange(0, 200), history.history['loss'], label="train_loss")
plt.plot(np.arange(0, 200), history.history['val_loss'], label="test_loss")
plt.legend()
plt.show()
