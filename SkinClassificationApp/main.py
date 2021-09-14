from typing import Optional
from fastapi import FastAPI, Request, Depends, BackgroundTasks,  File, UploadFile
from fastapi.templating import Jinja2Templates
import os
from fastapi.responses import HTMLResponse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
import cv2
import models
from database import SessionLocal, engine
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models import Client
import numpy as np
import boto3
#import itertool
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16

# include top should be False to remove the softmax layer
pretrained_model = VGG16(include_top=False, weights='imagenet')
app = FastAPI()
models.Base.metadata.create_all(bind=engine)
tempplates = Jinja2Templates(directory="templetes")
model = None

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.get("/")
def read_root(request: Request, db: Session = Depends(get_db)):
    patients = db.query(Client).all()
    global model
    if model is None:
        model = load_model("model_mine.h5")

    return tempplates.TemplateResponse("dashboard.html",{
        "request":request,
        "patients":patients
    })



def predict_the_data(wavfile):

    content = wavfile.file
    # load model
    contents = wavfile.file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    
    #image= cv2.imread( img, cv2.COLOR_BGR2RGB)
    image=cv2.resize(img, (256, 256),interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
   
   
    x_test = []
    Fpred="nema"
    Fconf=0
    #x_test.append(extract_features(classify_file, 60))
    x_test.append(image)
    x_test = np.asarray(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
    print("DIMENZII",x_test.shape)
    vgg_features_test = pretrained_model.predict(x_test)
    print("VGG shape",vgg_features_test.shape)
    pred = model.predict(vgg_features_test, verbose=1)
    print('all preds',pred)
    Fconf = np.round((max(pred[0]) * 100),2)
    true_class=np.argmax(pred,axis=1)
    print("true class", true_class,"conf:", Fconf)
    if true_class[0] == 0:
        Fpred = "Actinic keratosis"
       
    elif (true_class[0] == 1):
        Fpred = "Melanocytic nevus"
       
    elif (true_class[0] == 2):
        Fpred = "Basal cell carcinoma"
        
    elif (true_class[0] == 3):
        Fpred = "Dermatofibroma"
      
    elif (true_class[0] == 4):
        Fpred = "Melanoma"
       
    elif (true_class[0] == 5):
        Fpred = "Benign keratosis"
        
    elif (true_class[0] == 6):
        Fpred = "Squamous cell carcinoma"
       
    else:
        Fpred = "Vascular lesion"
        

    celConf = [round((pred[0][0] * 100), 2), round((pred[0][1] * 100), 2), round((pred[0][2]*100), 2), round((pred[0][3]*100), 2), round((pred[0][4]*100), 2), round((pred[0][5]*100), 2), round((pred[0][6]*100), 2), round((pred[0][7]*100), 2)]
    return Fpred, Fconf, celConf


@app.post("/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile = File(...),  db: Session = Depends(get_db)):

    result1, conf, result2 = predict_the_data(file)
    with open("test.mp3", 'wb') as f:
        f.write(file.file.read())
    clasa = Client()
    clasa.prediction = result1
    clasa.confidenceLevel = conf
    clasa.confidenceLevel1 = result2[0]
    clasa.confidenceLevel2 = result2[1]
    clasa.confidenceLevel3 = result2[2]
    clasa.confidenceLevel4 = result2[3]
    clasa.confidenceLevel5 = result2[4]
    clasa.confidenceLevel6 = result2[5]
    clasa.confidenceLevel7 = result2[6]
    clasa.confidenceLevel8 = result2[7]


    db.add(clasa)
    db.commit()
    patients = db.query(Client).all()

    #upload_new_file(patients[-1].id, patients[-1].prediction)

    return tempplates.TemplateResponse("dashboardPost.html", {
        "request": request,
        "patients": patients
    })
