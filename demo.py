"""
Created by Deng Jiali
charlesdjl@qq.com
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import time
from astropy.convolution import convolve, Gaussian1DKernel

begin = time.time()
rawData = pd.read_csv("./RawInputData.csv", encoding='utf-8')
rawData = rawData[['trade_date',"close","open","high","low",'RSI']]

# 抛弃含有缺失值的数据：由于缺失的值是由于前期数据用于取平均值，插值法不适用
rawData.drop(rawData.head(27).index, inplace=True)

day = 70
# 对数据集除了收盘价的参量进行高斯核平滑

open = rawData['open']
close = rawData["close"]
high = rawData['high']
low = rawData['low']
rsi = rawData['RSI']

gauss_kernel1 = Gaussian1DKernel(0.2) # 价格
gauss_kernel2 = Gaussian1DKernel(1) # RSI

smoothed_open = convolve(open, gauss_kernel1)
smoothed_high = convolve(high, gauss_kernel1)
smoothed_low = convolve(low, gauss_kernel1)
smoothed_rsi = convolve(rsi, gauss_kernel2)

rawData['open'] = smoothed_open
rawData['close'] = smoothed_close
rawData['high'] = smoothed_high
rawData['low'] = smoothed_low
rawData['RSI'] = smoothed_rsi

# 划分数据集。共有2603条有效数据，将前2000条数据作为训练集，后面603条数据作为测试集。比例为77：23.
timeIndex = rawData['trade_date']
train_set = rawData.iloc[0:2000, 1:]
test_set = rawData.iloc[2000:, 1:]

# 数据标准化：最大-最小规范化,归一化。对训练集及测试集分别进行
# fit用于求最大值最小值均值等属性，在测试的时候假设这种属性也存在于测试集，故一般训练集需要fit和transform，而测试集只需要transform。
scaler = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)

# 利用for循环构建数据集和数据标签。每连续60天的数据作为输入，第61天的收盘价作为标签。
# 由此，每个输入矩阵为60*9。训练集共2000-60=1940组数据，测试集共543组
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(day, len(train_set_scaled)):
    x_train.append(train_set_scaled[i-day : i])
    # 收盘价位于数据集第1列
    y_train.append(train_set_scaled[i,0:1])

for i in range(day, len(test_set)):
    x_test.append(test_set[i-day : i])
    y_test.append(test_set[i,0:1])

# 打乱训练集，并将数据转换为数组形式
np.random.seed(1)
np.random.shuffle(x_train)
np.random.seed(1)
np.random.shuffle(y_train)
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
# 增加一维的通道数，以输入卷积层。
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

class myCNN(Model):
    def __init__(self):
        super(myCNN, self).__init__()
        self.c1 = Conv2D(filters=8, kernel_size=(4, 4), padding="same")
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=1)
        self.c2 = Conv2D(filters=16, kernel_size=(4, 4), padding="same")
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=1)
        self.c3 = Conv2D(filters=32, kernel_size=(4, 4), padding="same")
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=1)
        self.c4 = Conv2D(filters=64, kernel_size=(4, 4), padding="same")
        self.flatten = Flatten()
        self.d1 = Dropout(0.2)
        self.f1 = Dense(1)
    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.c3(x)
        x = self.p3(x)
        x = self.c4(x)
        x = self.flatten(x)
        x = self.d1(x)
        y = self.f1(x)
        return y
model = myCNN()
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='mean_squared_error')
              # metrics=['mae'])
history = model.fit(x_train, y_train, epochs=120,#batch_size=32,
                    validation_data=(x_test, y_test),
                    validation_freq=1) #,
                    #callbacks=[cp_callback])
model.summary()

# 作图
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

################## predict ######################
# 测试集输入模型进行预测
predicted_price = model.predict(x_test)
# 删除测试集中的收盘价，并将预测值并入数据中
temp_set = np.column_stack((predicted_price,test_set[day:,1:]))
"""
temp_set = np.delete(test_set,0,axis = 1)
temp_set = temp_set[60:]
temp_set = np.column_stack((predicted_price,temp_set))
"""

# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_price = scaler.inverse_transform(temp_set)
real_stock_price = scaler.inverse_transform(test_set[day:])
plt.figure(2)
plt.plot(real_stock_price[:,0], color='red', label='Real SCI')
plt.plot(predicted_price[:,0], color='blue', label='Predicted SCI')
plt.title('SCI Prediction')
plt.xlabel('Time')
plt.ylabel('SCI')
plt.legend()
plt.show()

# 取最后60天数据观测
plt.figure(3)
plt.plot(real_stock_price[300:400, 0], color='red', label='Real SCI')
plt.plot(predicted_price[300:400, 0], color='blue', label='Predicted SCI')
plt.title('Latest 60 days')
plt.xlabel('Time')
plt.ylabel('SCI')
plt.legend()
plt.show()

# 评测指标的计算
y_true = real_stock_price[:,0]
y_pred = predicted_price[:,0]

# 1.计算相关系数
series1 = pd.Series(y_true)
series2 = pd.Series(y_pred)
corr = series1.corr(series2)
print("相关系数为：" + str(corr))

# 2.计算确定系数
r_squar = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
print("确定系数为：" + str(r_squar))

# 3.计算RMSE
RMSE = metrics.mean_squared_error(y_true, y_pred)**0.5
print("RMSE为：" + str(RMSE))
print("day = " + str(day))
end = time.time()
print("训练耗时：" + str(end-begin))
