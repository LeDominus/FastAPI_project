from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from models.inference.inference import predictor
from models.train.train import trainer

app = FastAPI()


@app.post('/train', tags=['Обучение'], summary='Ручка для обучения' )
def train_model():
    trainer.preprocess_data()
    return {"message": "Модель обучена и сохранена"}

@app.post('/predict', tags=['Прогноз'], summary='Ручка для прогнозирования')
def get_prediction():
    prediction = predictor.predict()
    return {'prediction': prediction.tolist()}

@app.post('/metrics', tags=['Метрики'], summary='Ручка для получения метрик')
def get_metrics():
    return predictor.metrics()

@app.get('/', tags=["Главный роут"])
def root():
    return {'message': 'Hello World!'}