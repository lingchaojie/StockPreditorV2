import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


plt.figure(figsize=(20,10))
np.random.seed(7)

def readStockData(file, path='./data/'):

    data = pd.read_csv(path+file)
    return data

dataset = readStockData('XBTUSD.csv')
dataset.head()
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Close'] = pd.to_numeric(dataset['Close'], downcast='float')
dataset.set_index('Date',inplace=True)

close = dataset['Close']
close = close.values.reshape(len(close), 1)
# plt.plot(close)
# plt.show()


series=7



def create_ts(ds, series):
    X, Y =[], []
    for i in range(len(ds)-series - 1):
        item = ds[i:(i+series), 0]
        X.append(item)
        Y.append(ds[i+series, 0])
    return np.array(X), np.array(Y)


def TrainAndTest(close, split, epochs, series):

    scaler = MinMaxScaler(feature_range=(0, 1))
    close = scaler.fit_transform(close)
    TrainData, TestData = close[0:split], close[split:]

    testX, testY = create_ts(TestData, series)

    trainX, trainY = create_ts(TrainData, series)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    model = Sequential()
    model.add(LSTM(16, return_sequences=True, input_shape=(series, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=False, input_shape=(series, 1)))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=30)

    trainPredictions = model.predict(trainX)
    testPredictions = model.predict(testX)

    trainPredictions = scaler.inverse_transform(trainPredictions)
    testPredictions = scaler.inverse_transform(testPredictions)

    trainY = scaler.inverse_transform([trainY])
    testY = scaler.inverse_transform([testY])

    train_plot = np.empty_like(close)
    train_plot[:, :] = np.nan
    train_plot[series:len(trainPredictions) + series, :] = trainPredictions

    test_plot = np.empty_like(close)
    test_plot[:, :] = np.nan
    test_plot[len(trainPredictions) + (series * 2) + 1:len(close) - 1, :] = testPredictions

    plt.plot(scaler.inverse_transform(close), color = 'blue', label = 'True Value')
    plt.plot(train_plot, color = 'orange', label = 'Train Value')
    plt.plot(test_plot, color = 'green', label = 'Predict Value')

    plt.legend()
    plt.show()

    trainError = math.sqrt(mean_squared_error(trainY[0], trainPredictions[:, 0]))
    testError = math.sqrt(mean_squared_error(testY[0], testPredictions[:, 0]))

    plt.text(0.5, 0.5, 'matplotlib', horizontalalignment='center', verticalalignment = 'center')





def Predict(times,close,epochs, series):

    length = len(close)


    scaler = MinMaxScaler(feature_range=(0, 1))

    close = scaler.fit_transform(close)
    copy_close = close.copy()


    TrainData = close[:]



    trainX, trainY = create_ts(TrainData, series)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # print('X:::::')
    # print(trainX[1])
    #
    # print('Y:::::')
    # print(trainY[1])

    model = Sequential()
    model.add(LSTM(16, return_sequences=True, input_shape=(series, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=False, input_shape=(series, 1)))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=30)




    trainPredictions = model.predict(trainX)



    trainY = scaler.inverse_transform([trainY])

    print('Y的shape：', trainY.shape)

    trainPredictions = scaler.inverse_transform(trainPredictions)

    train_plot = np.empty_like(close)
    train_plot[:,:] = np.nan
    train_plot[series-1:len(close)-2, : ] = trainPredictions

    print('length of TR', len(train_plot))

    predict_plot = np.empty_like(copy_close)
    predict_plot[:, :] = np.nan


    print('copySHAPE')
    print(copy_close.shape)
    for i in range(times):

        future = copy_close[-series:]
        print('future: ', future)
        futureData = np.transpose(future)
        futureData = np.reshape(futureData, (futureData.shape[0], futureData.shape[1], 1))
        futurePredict = model.predict(futureData)
        copy_close = np.append(copy_close, [futurePredict])
        copy_close = copy_close.reshape(len(copy_close),1)
        futurePredict = scaler.inverse_transform(futurePredict)

        predict_plot = np.append(predict_plot,futurePredict)






    print('predict:' ,predict_plot.shape)
    print('copy: ' ,copy_close.shape)

    print('-0-----------------------------')
    print(predict_plot)
    # print(copy_close[len(close):])

    # for i in range(times):
    #     # predict_plot[len(close)-2] = scaler.inverse_transform(copy_close[len(close)+i])
    #     copy_close[len(close)+i] =

    plt.plot(predict_plot, color = 'red')
    plt.plot(train_plot,color = 'orange')



    plt.plot(scaler.inverse_transform(close), color = 'blue')

    plt.show()








# _, _ = TrainAndTest(close, 220, 600, series)
Predict(20,close,60, series)




#testX, testY = create_ts(TestData, series)

# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# model = Sequential()
# model.add(LSTM(16, return_sequences=True, input_shape=(series, 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(30, return_sequences=False, input_shape=(series, 1)))
# model.add(Dense(1))
# model.add(Activation('linear'))
#
#
# model.compile(loss='mse', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=30)


# print('close shape: ',close.shape)
#
#
# trainPredictions = model.predict(trainX)
# trainY = scaler.inverse_transform([trainY])
#
#
# trainPredictions = scaler.inverse_transform(trainPredictions)
#
#
# train_plot = np.empty_like(close)
# train_plot[:,:] = np.nan
# train_plot[series+1:len(close), : ] = trainPredictions
#
# print('prediction shape: ',trainPredictions.shape)
# print('predict starts from: ', series+1, 'to ', len(close))
#
#
#
#
#
#
#
# #Future
# future = close[-series:]
#
#
# futureData = np.transpose(future)
#
#
# futureData = np.reshape(futureData, (futureData.shape[0], futureData.shape[1], 1))
# futurePredict = model.predict(futureData)
# futurePredict = scaler.inverse_transform(futurePredict)
#
#
#
#
#
# plt.scatter(len(close),futurePredict,color = 'red')
# plt.plot(train_plot,color = 'orange')
#
#
#
# plt.plot(scaler.inverse_transform(close), color = 'blue')

# plt.plot(test_plot)
#
# plt.show()


# train_plot[series:len(trainPredictions)+series, :] = trainPredictions

'''
print(trainPredictions)
print(len(trainPredictions))
testPredictions = model.predict(testX)

trainPredictions = scaler.inverse_transform(trainPredictions)
testPredictions = scaler.inverse_transform(testPredictions)

trainY = scaler.inverse_transform([trainY])
testY = scaler.inverse_transform([testY])


train_plot = np.empty_like(close)
train_plot[:,:] = np.nan
train_plot[series:len(trainPredictions)+series, :] = trainPredictions

test_plot = np.empty_like(close)
test_plot[:,:] = np.nan
test_plot[len(trainPredictions)+(series*2)+1:len(close)-1, :] = testPredictions

# plt.plot(scaler.inverse_transform(close))
# plt.plot(train_plot)
# plt.plot(test_plot)
# 
# plt.show()
'''

#
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
#
# trainX, trainY = create_ts(TrainData, series=10)
# testX, testY = create_ts(TestData, series=10)
#
# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))
#
#
# print (trainX.shape, trainY.shape)
# print(testX.shape, testY.shape)
#
#
# kernel = Matern(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
# gpr.fit(trainX, trainY)
#
# trainPredictions = gpr.predict(trainX)
# testPredictions = gpr.predict(testX)
#
#
# trainPredictions = np.reshape(trainPredictions, [212,1])
# testPredictions = np.reshape(testPredictions, [24,1])
#
# print(trainPredictions.shape)
# print(testPredictions.shape)
#
#
# trainPredictions = scaler.inverse_transform(trainPredictions)
# testPredictions = scaler.inverse_transform(testPredictions)
#
# trainY = np.reshape(trainY, [212,1])
# testY = np.reshape(testY, [24,1])
# trainY = scaler.inverse_transform(trainY)
# testY = scaler.inverse_transform(testY)
#
# train_plot = np.empty_like(close)
# train_plot[:,:] = np.nan
# train_plot[10:len(trainPredictions)+series, :] = trainPredictions
#
#
# test_plot = np.empty_like(close)
# test_plot[:,:] = np.nan
# test_plot[len(trainPredictions)+(series*2)+1:len(close)-1, :] = testPredictions
#
#
# #future_plot = np.empty_like(close)
# #test_plot[:,:] = np.nan
#
# plt.plot(scaler.inverse_transform(close))
# plt.plot(train_plot)
# plt.plot(test_plot)
#
# plt.show()