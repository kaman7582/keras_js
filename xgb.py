from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
all_data=read_csv("data/data.csv")

val = list(all_data['C2H2'])

'''

all_data=read_csv("data/data_1.csv")

val = list(all_data['1#'])


def preprocess_data(dataset,history):
    data_x = []
    data_y = []
    for i in range(len(dataset)-history):
        data_x.append(dataset[i:i+history])
        data_y.append(dataset[i+history])
    return np.array(data_x),np.array(data_y)


def raw_to_supervised(dataset,history,output_days):
    df = DataFrame(dataset)
    col = list()
    for i in range(history,0,-1):
        col.append(df.shift(i))
    #print(col)
    for i in range(0, output_days):
        col.append(df.shift(-i))
    agg = concat(col, axis=1)
    agg.dropna(inplace=True)
    return (agg)

def train_validation(data, n_test):
    predictions = list()
    # 分割数据集
    train, test = train_test_split(data,test_size= 0.7)


trainx,trainy = preprocess_data(val,5)

X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.3, random_state=0)

param = {'max_depth':5, 'eta':0.1, 'silent':0, 'subsample':0.7, 'colsample_bytree':0.7, 'objective':'binary:logistic' }
num_round = 10

dtrain = xgb.DMatrix(X_train, label=y_train)

model = XGBRegressor(n_estimators=3000,
                     max_depth=10,
                     colsample_bytree=0.5, 
                     subsample=0.5, 
                     learning_rate = 0.01
                    )

res = model.fit(X_train, y_train, verbose=True)
#testx = [[0.1,0.2,0.3,0.4,0.3]]
test_rest = model.predict(X_test)
plt.plot(test_rest,color='red')
plt.plot(y_test,color='blue')
plt.show()
#eval_rst={}
#xgb.train(params,dtrain,num_boost_round=20,evals_result=eval_rst,verbose_eval=True)
#print(eval_rst)
#bst = xgb.train(param, dtrain, num_round, watchlist, logregobj, evalerror)

#dtrain = xgb.DMatrix(trainx, trainy)

#xgb.cv(param, dtrain, num_round, nfold=5,metrics={'error'}, seed = 0)

'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        #print("-----------------")
    #print(cols)
    out = concat(cols, axis=1)
    #print(out.dropna())
    #return 
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # 合并到一起
    agg = concat(cols, axis=1)
    #丢弃含有NaN的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

val=[[1],[2],[3],[4],[5]]

series_to_supervised(val,3,1)

'''
