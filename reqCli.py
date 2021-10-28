from pydantic.types import Json
import requests
from pandas import read_csv
import json
import numpy as np
#half_plus_two
#192.168.0.180 port 8501
'''
uri=http://192.168.0.180:8501/v1/models/half_plus_two:predict
query_data = '{"instances": [[1.0, 2.0, 3.0]}'
requests.post(url, query_data)
'''

'''
curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://192.168.0.180:8501/v1/models/half_plus_two:predict

curl -XPOST http://192.168.0.180:8501/v1/models/half_plus_two:predict -d "{\"instances\":[1.0, 2.0, 5.0]}"

tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} "$@"


'''
all_data=read_csv("data/data_1.csv")

val = list(all_data['1#'])

history = val[-5:]

history = np.array(history).reshape(-1,1)


payload = {
    "instances": [history.tolist()]
}
print(payload)
req_data = json.dumps(payload)

print(req_data)

def send_test():
    url="http://192.168.0.180:8501/v1/models/c2h2:predict"
    query_data = {"instances": [[[0.1]]]}
    res =requests.post(url, json=payload)
    print(json.loads(res.content.decode('utf-8')))


send_test()

'''

all_data=read_csv("data/data.csv")

val = list(all_data['C2H2'])

json_msg={}
json_msg['data']=val
json_msg['gas_type']='c2h2'
#fpp = json.dump(data1)
#data =  [{ 'gas_type' : 'c2h2', 'data' : '1'}]
 
#data2 = json.dumps(data)

data = {
    'phone': 1,
    'code': 2,
}


url = "http://127.0.0.1:8000/file"
files = {'abc': open('data/data.csv', 'rb')}
#res = requests.post(url, files=files)
#url = "http://127.0.0.1:8000/login"
#res = requests.post(url, json=data)

#url = "http://127.0.0.1:8000/train"
#res = requests.post(url, json=json_msg)

url = "http://127.0.0.1:8000/predict"
res = requests.post(url, json=json_msg)

#print(res)
print(res.json())

'''