from matplotlib import colors
import tensorflowjs as tfjs
import tensorflow as tf
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from numpy import concatenate
import numpy as np
from math import sqrt
import os

epochs=1000
SAMPLE_COL = 5
look_back = 5
train_percent=0.7
col_name=['H2','CH4','C2H4','C2H6','C2H2']
trans_data=[]

raw_data_path="./data/data.csv"
save_module_path="./models/"

def create_data(dataset):
    data_x = []
    data_y = []
    for i in range(len(dataset)-look_back):
        data_x.append(dataset[i:i+look_back])
        data_y.append(dataset[i+look_back])
    return data_x,data_y #转为ndarray数据

def dataset_parser():
    all_data=read_csv(raw_data_path)
    # get all the information
    for i in col_name:
        tmp = list(all_data[i])
        trans_data.append(tmp)
    #convert all 
    idx = col_name.index('C2H2')
    tmpx,tmpy = create_data(trans_data[idx])
    return tmpx,tmpy

datax,datay = dataset_parser()
datax = np.array(datax)
datay = np.array(datay)
datax = datax.reshape(datax.shape[0],datax.shape[1],1)
total_len = len(datax)
train_len = int(total_len * train_percent)
#trainx = np.array(trainx,dtype=float)

#data normalize

# reshape input to be 3D [samples, timesteps, features]

trainx = datax[:train_len,:,:]
trainy = datay[:train_len]

testx = datax[train_len:,:,:]
testy = datay[train_len:]
#LSTM(CELL_SIZE, input_shape = (TIME_STEPS,INPUT_SIZE))
if os.path.exists('./models/c2h2_analysis.model') == False:
    model = Sequential()
    model.add(LSTM(50, input_shape=(trainx.shape[1], trainx.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(trainx,trainy,epochs=100,batch_size=50,verbose=2,shuffle=False)
    model.save('./models/c2h2_analysis.model')
else:
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
# 显示图例：
#pyplot.legend()
#pyplot.show()
    model = tf.keras.models.load_model('./models/c2h2_analysis.model')
    yhat = model.predict(datax)
    plt.plot(datax[:,0],color='red')
    plt.plot(yhat[:,0],color='blue')
    plt.show()