import pandas as pd
import numpy as np
import yaml
import os
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(data_path, target_path):
    """Загрузка обработанных данных и целевых переменных"""
    logger.info(f"Загрузка обработанных данных из {data_path}")
    X = pd.read_csv(data_path)
    y = pd.read_csv(target_path)['target']
    return X, y

def get_model(model_name, model_params):
    """Инициализация модели на основе конфигурации"""
    logger.info(f"Инициализация модели {model_name}")
    
    if model_name.lower() == 'random_forest':
        return RandomForestClassifier(**model_params)
    elif model_name.lower() == 'logistic_regression':
        return LogisticRegression(**model_params)
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_name}")

def train_model(model, X_train, y_train):
    """Обучение модели"""
    logger.info("Обучение модели")
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Сохранение обученной модели на диск"""
    logger.info(f"Сохранение модели в {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def evaluate_model(model, X_test, y_test):
    """Оценка производительности модели"""
    logger.info("Оценка модели")
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics

def save_metrics(metrics, metrics_path):
    """Сохранение метрик в JSON файл"""
    logger.info(f"Сохранение метрик в {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    """Главная функция для обучения модели"""
    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Для обратной совместимости будем использовать config для всех параметров
    params = config
    
    try:
        # Загрузка обработанных данных
        train_data_path = os.path.join(config['data']['processed_path'], config['data']['train_file'])
        train_target_path = train_data_path.replace('.csv', '_target.csv')
        
        X_train, y_train = load_processed_data(train_data_path, train_target_path)
        
        # Инициализация модели
        model = get_model(params['model']['name'], params['model']['params'])
        
        # Обучение модели
        trained_model = train_model(model, X_train, y_train)
        
        # Сохранение модели
        model_path = "models/model.pkl"
        save_model(trained_model, model_path)
        
        # Оценка на обучающем наборе (для демонстрации)
        train_metrics = evaluate_model(trained_model, X_train, y_train)
        logger.info(f"Метрики обучения: {train_metrics}")
        
        # Сохранение метрик
        save_metrics(train_metrics, "metrics.json")
        
        logger.info("Обучение модели успешно завершено")
        
    except Exception as e:
        logger.error(f"Ошибка в обучении модели: {str(e)}")
        raise

if __name__ == "__main__":
    main()
