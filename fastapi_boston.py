import numpy as np
import pandas as pd
from fastapi import FastAPI
import joblib
from pydantic import BaseModel,Field
import sqlite3
import json
from datetime import datetime

app=FastAPI()
model=joblib.load("boston_house.pkl")

#create database
conn=sqlite3.connect("Prediction_Boston.db",check_same_thread=False)
cursor=conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS Boston_House_Predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input TEXT,
                prediction REAL,
                created_at Text)''')
conn.commit()

class HousePrice(BaseModel):
    CRIM:float=Field(...,example=0.00632)
    INDUS:float=Field(...,example=18)
    NOX:float=Field(...,example=0.538)
    RM:float=Field(...,example=6.575)
    AGE:float=Field(...,example=65.2)
    DIS:float=Field(...,example=4.0900)
    RAD:int=Field(...,example=1)
    TAX:float=Field(...,example=296.0)
    PTRATIO:float=Field(...,example=15.3)
    B:float=Field(...,example=396.90)
    LSTAT:float=Field(...,example=4.98)

@app.get("/")
def home():
    return {"message":"Welcome to the Boston House Price Prediction Dockerised ML Pipeline!"}

@app.post("/predict")
def predict(data:HousePrice):
    df=pd.DataFrame([data.model_dump()])
    pred=model.predict(df)[0]
    #insert into table
    cursor.execute('''INSERT INTO Boston_House_Predictions (input,prediction,created_at) VALUES (?,?,?)''',  (json.dumps(data.model_dump()),
                                                                                                              float(pred),
                                                                                                              datetime.now().isoformat()))
    conn.commit()
    return {"Predicted price":round(float(pred),2)}

@app.get("/get_logs")
def get_logs():
    cursor.execute("SELECT * FROM Boston_House_Predictions ORDER BY id DESC")
    rows=cursor.fetchall()
    results=[]
    for row in rows:
        results.append({
            "id":row[0],
            "input":json.loads(row[1]),
            "prediction":row[2],
            "created_at":row[3]
        })
    return results

@app.get("/sample_input")
def sample_input():
    return {
        "CRIM":0.00632,
        "INDUS":18,
        "NOX":0.538,
        "RM":6.575,
        "AGE":65.2,
        "DIS":4.0900,
        "RAD":1,
        "TAX":296.0,
        "PTRATIO":15.3,
        "B":396.90,
        "LSTAT":4.98
    }
    
    