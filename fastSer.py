from fastapi import FastAPI, File, UploadFile
import uvicorn
import datetime
from pydantic import BaseModel
from util.trainset import gasTrain

app = FastAPI()
gasModel = gasTrain()

def start_train(dataset,name):
    if(len(dataset) > 0):
        gasModel.train_data(dataset,name)


#predict_gas(300)

class LoginUser(BaseModel):
    phone: str
    code: str

class trainSet(BaseModel):
    gas_type : str
    data:list

@app.post('/predict')
def predict_gas(predicts:trainSet):
    datas = list(predicts.data)
    result = gasModel.predict_data(datas,predicts.gas_type)
    print("predict",result)

@app.post('/train')
def user_login(trains: trainSet):
    #print(trains.data)
    #print(trains.gas_type)
    datas = list(trains.data)
    start_train(datas,trains.gas_type)
    return {'msg':'data got'}


@app.post('/login')
def user_login(user: LoginUser):
    # 查询phone 是否存在
    # 验证code 是否有效
    return {'msg':'用户已登录', 'phone':user.phone,'number':user.code}

def parser_content(content):
    print(type(content))
    print(content)

@app.post('/file')
async def _file_upload(abc: UploadFile = File(...)):
    content = await abc.read()
    #all_data=read_csv(abc)
    #print(all_data)
    parser_content(content)
    return {"filename": abc.filename}

'''
@app.post('/file')
def _file_upload(my_file: UploadFile = File(...)):
    print(my_file.filename)
    #print(my_file.read(100))
'''


@app.get("/now")
async def get_time():
   time = datetime.datetime.now()
   return time.strftime("%Y-%m-%d %H:%M:%S")

@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="debug")
