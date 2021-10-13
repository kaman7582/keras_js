from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.random.ranf(10)
data = data.reshape(-1,1)
mm = MinMaxScaler()
print(data)
print("----------------------------------------")
mm_data = mm.fit_transform(data)
print(mm_data)
print("----------------------------------------")
origin_data = mm.inverse_transform(mm_data)
print(origin_data)
print("----------------------------------------")
