from util.trainset import gasTrain
from pandas import read_csv


all_data=read_csv("data/data_1.csv")

val = list(all_data['1#'])
val_len = len(val)

gas_model = gasTrain(future_day=3)

result = gas_model.predict_future(val,"c2h2_sta1")

if len(result)  > 0:
    gas_model.predict_display(val,result)