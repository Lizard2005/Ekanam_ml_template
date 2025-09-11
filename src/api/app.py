import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import yaml
from typing import List, Dict, Any

# Инициализация FastAPI приложения
app = FastAPI(
    title="ML Model API",
    description="API для предоставления прогнозов модели машинного обучения",
    version="0.1.0"
)

# Глобальные переменные для модели и конфигурации
model = None
config = None

class PredictionRequest(BaseModel):
    """Модель запроса для эндпоинта прогнозирования"""
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    """Модель ответа для эндпоинта прогнозирования"""
    prediction: int
    probability: float

def load_model_and_config():
    """Загрузка обученной модели и конфигурации"""
    global model, config
    
    # Загрузка конфигурации
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Загрузка модели
    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print(f"Предупреждение: Файл модели {model_path} не найден")

@app.on_event("startup")
async def startup_event():
    """Загрузка модели и конфигурации при запуске"""
    load_model_and_config()

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "message": "ML Model API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Эндпоинт проверки состояния"""
    model_status = "загружена" if model is not None else "не загружена"
    return {
        "status": "здоровая",
        "model_status": model_status
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Сделать прогноз с использованием обученной модели"""
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    try:
        # Преобразование признаков в DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Сделать прогноз
        prediction = model.predict(features_df)[0]
        
        # Получить вероятность прогноза, если доступна
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_df)[0]
            probability = float(np.max(probabilities))
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка прогнозирования: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Получить информацию о модели"""
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    info = {
        "model_type": model.__class__.__name__,
        "features": []
    }
    
    # Попытка получить имена признаков, если доступны
    if hasattr(model, "feature_names_in_"):
        info["features"] = model.feature_names_in_.tolist()
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
