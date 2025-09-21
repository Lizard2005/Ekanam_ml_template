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
import boto3
from io import StringIO
import mlflow
import mlflow.sklearn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data_from_s3(bucket_name, data_key, target_key, aws_access_key_id=None, aws_secret_access_key=None):
    """Загрузка обработанных данных и целевых переменных из S3"""
    logger.info(f"Загрузка обработанных данных из s3://{bucket_name}/{data_key} и s3://{bucket_name}/{target_key}")
    
    # Создание клиента S3
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client('s3', 
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)
    else:
        # Используем IAM роли или credentials из окружения
        s3 = boto3.client('s3')
    
    try:
        # Загрузка данных
        data_response = s3.get_object(Bucket=bucket_name, Key=data_key)
        data_content = data_response['Body'].read().decode('utf-8')
        X = pd.read_csv(StringIO(data_content))
        
        # Загрузка целевых переменных
        target_response = s3.get_object(Bucket=bucket_name, Key=target_key)
        target_content = target_response['Body'].read().decode('utf-8')
        y = pd.read_csv(StringIO(target_content))['target']
        
        return X, y
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из S3: {str(e)}")
        raise

def load_processed_data(data_path, target_path):
    """Загрузка обработанных данных и целевых переменных (локально)"""
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

def upload_to_s3(local_file_path, bucket_name, s3_key, aws_access_key_id=None, aws_secret_access_key=None):
    """Загрузка файла в S3"""
    logger.info(f"Загрузка {local_file_path} в s3://{bucket_name}/{s3_key}")
    
    # Создание клиента S3
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client('s3', 
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)
    else:
        # Используем IAM роли или credentials из окружения
        s3 = boto3.client('s3')
    
    # Загрузка файла
    try:
        s3.upload_file(local_file_path, bucket_name, s3_key)
        logger.info(f"Файл успешно загружен в S3: s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла в S3: {str(e)}")
        raise

def main():
    """Главная функция для обучения модели"""
    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Для обратной совместимости будем использовать config для всех параметров
    params = config
    
    # Параметры S3 из переменных окружения
    s3_bucket = os.getenv('S3_BUCKET')
    s3_processed_prefix = os.getenv('S3_PROCESSED_PREFIX', 'data/processed/')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # Параметры MLflow из переменных окружения
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow_experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'ml-project')
    
    # Настройка MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    
    try:
        # Загрузка обработанных данных из S3 или локально
        train_data_path = os.path.join(config['data']['processed_path'], config['data']['train_file'])
        train_target_path = train_data_path.replace('.csv', '_target.csv')
        
        if s3_bucket:
            X_train, y_train = load_processed_data_from_s3(
                s3_bucket, 
                s3_processed_prefix + config['data']['train_file'],
                s3_processed_prefix + config['data']['train_file'].replace('.csv', '_target.csv'),
                aws_access_key_id, 
                aws_secret_access_key
            )
        else:
            X_train, y_train = load_processed_data(train_data_path, train_target_path)
        
        # Начало эксперимента MLflow
        with mlflow.start_run():
            # Логирование параметров
            mlflow.log_params({
                'model_name': params['model']['name'],
                'test_size': params['data']['test_size'],
                'random_state': params['data']['random_state']
            })
            
            # Логирование гиперпараметров модели
            for param, value in params['model']['params'].items():
                mlflow.log_param(f"model_{param}", value)
            
            # Инициализация модели
            model = get_model(params['model']['name'], params['model']['params'])
            
            # Обучение модели
            trained_model = train_model(model, X_train, y_train)
            
            # Сохранение модели
            model_path = "models/model.pkl"
            save_model(trained_model, model_path)
            
            # Загрузка модели в S3, если указан bucket
            if s3_bucket:
                upload_to_s3(model_path, s3_bucket, 'models/model.pkl', 
                             aws_access_key_id, aws_secret_access_key)
            
            # Оценка на обучающем наборе (для демонстрации)
            train_metrics = evaluate_model(trained_model, X_train, y_train)
            logger.info(f"Метрики обучения: {train_metrics}")
            
            # Логирование метрик в MLflow
            mlflow.log_metrics(train_metrics)
            
            # Сохранение метрик локально
            save_metrics(train_metrics, "metrics.json")
            
            # Логирование артефактов модели в MLflow
            mlflow.sklearn.log_model(trained_model, "model")
            
            # Логирование файла метрик как артефакт
            mlflow.log_artifact("metrics.json")
            
        logger.info("Обучение модели успешно завершено")
        
    except Exception as e:
        logger.error(f"Ошибка в обучении модели: {str(e)}")
        raise

if __name__ == "__main__":
    main()
