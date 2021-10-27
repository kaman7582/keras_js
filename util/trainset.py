# -*- coding: utf-8 -*-
# file: trainset.py
# Copyright (C) 2021. All Rights Reserved.

from matplotlib import colors
import tensorflowjs as tfjs
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

#train the feature data
class gasTrain():
    def __init__(self,name='unknown',epoch=500,step=5,future_day=5) -> None:
        self.epoch = epoch
        self.history_days = step
        self.model_name = name
        self.predict_days = future_day
    
    #create the data range
    def preprocess_data(self,dataset):
        data_x = []
        data_y = []
        for i in range(len(dataset)-self.history_days):
            data_x.append(dataset[i:i+self.history_days])
            data_y.append(dataset[i+self.history_days])
        return np.array(data_x),np.array(data_y)

    def standard_data(self,data):
        scaler = MinMaxScaler()
        raw_data = scaler.fit_transform(data)
        return raw_data

    def train_data(self,raw_data,data_name):
        self.model_name = './models/{}.model'.format(data_name)
        #normalized data
        scaler = MinMaxScaler()
        raw_data=np.array(raw_data).reshape(-1, 1)
        raw_data = scaler.fit_transform(raw_data)
        
        datax,datay = self.preprocess_data(raw_data)
        datax = datax.reshape(datax.shape[0],datax.shape[1],1)
        #define lstm model
        model = Sequential()
        model.add(LSTM(50, input_shape=(datax.shape[1], datax.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(datax,datay,epochs=self.epoch,batch_size=50,verbose=2 ,shuffle=False)
        model.save(self.model_name)

    def predict_future(self,raw_data,data_name):
        '''
        #raw_data += list()
        predict_data= []
        future_data =np.zeros(self.predict_data)
        old_data = np.array(raw_data[-self.step:])
        mix_data = old_data + future_data

        if os.path.exists(self.model_name) == True:
            model = load_model(self.model_name)
            for i in range(self.predict_data):
                his_data = mix_data[i:self.step+i]
                pre_data = his_data.reshape(1,self.step,1)
                result = model.predict(pre_data)
                his_data[self.step+i] = result
                predict_data.append(result)
                #return {'predict':result}
        else:
            return {'predict':'Error'}
        '''
        #standard
        self.model_name = './models/{}.model'.format(data_name)
        scaler = MinMaxScaler()
        raw_data=np.array(raw_data).reshape(-1, 1)
        raw_data = scaler.fit_transform(raw_data)
        last_days = np.append(np.array(raw_data[-self.history_days:].copy()),np.zeros(self.predict_days))
        if os.path.exists(self.model_name) == True:
            model = load_model(self.model_name)
            #using history days like 5 days to predict next 3 days
            for i in range(self.predict_days):
                old_days = last_days[i:self.history_days+i]
                old_days = old_days.reshape(1,self.history_days,1)
                result = model.predict(old_days)
                last_days[self.history_days+i] = result
            #parser the result
            predict_data = last_days[self.history_days:]
            predict_data = predict_data.reshape(-1,1)
            predict_data = scaler.inverse_transform(predict_data)
            return predict_data
        else:
            return None
        #future_data =np.zeros(self.predict_days)
        #future_data = np.append(last_days,future_data)
    def predict_display(self,raw_data,future_data):
        old_len = len(raw_data)
        new_len = len(future_data)
        x_old=range(old_len)
        x_future=range(old_len,old_len+new_len)
        plt.plot(x_old,raw_data,color='blue')
        plt.plot(x_future,future_data,color='red')
        plt.show()


if __name__ == "__main__":
    from pandas import read_csv
    all_data=read_csv("data/data.csv")
    days=100
    cols = list(all_data['C2H2'])

    model = load_model("models/c2h2.model")
    #predict_list=np.zeros((300,5,1))
    #result = model.predict(predict_list)
    #print(result)
    val = cols[-5:]
    future_data =np.zeros(days)
    val =np.append(val,future_data)
    predict_list=[]
    for i in range(days):
        pred_data = val[i:5+i]
        pred_data = pred_data.reshape(1,5,1)
        result = model.predict(pred_data)
        predict_list.append(result[0][0])
        val[5 +i] = result
    #print(predict_list)
    test_val = predict_list[:days]
    x=range(len(cols))
    x1 = range(len(cols),len(cols)+days)
    print(x)
    print(len(x1),len(test_val))
    plt.plot(x,cols,color='red')
    plt.plot(x1,test_val,color='blue')
    plt.show()