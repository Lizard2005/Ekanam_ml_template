import pandas as pd
import numpy as np
import yaml
import os
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_model_from_s3(bucket_name, model_key, aws_access_key_id=None, aws_secret_access_key=None):
    """Загрузка обученной модели из S3"""
    logger.info(f"Загрузка модели из s3://{bucket_name}/{model_key}")
    
    # Создание клиента S3
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client('s3', 
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)
    else:
        # Используем IAM роли или credentials из окружения
        s3 = boto3.client('s3')
    
    try:
        # Загрузка модели
        model_response = s3.get_object(Bucket=bucket_name, Key=model_key)
        model_content = model_response['Body'].read()
        
        # Сохранение модели во временный файл
        temp_model_path = "temp_model.pkl"
        with open(temp_model_path, 'wb') as f:
            f.write(model_content)
        
        # Загрузка модели
        model = joblib.load(temp_model_path)
        
        # Удаление временного файла
        os.remove(temp_model_path)
        
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели из S3: {str(e)}")
        raise

def load_model(model_path):
    """Загрузка обученной модели с диска"""
    logger.info(f"Загрузка модели из {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    """Оценка производительности модели"""
    logger.info("Оценка модели")
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
    }
    
    return metrics, y_pred

def save_metrics(metrics, metrics_path):
    """Сохранение метрик в JSON файл"""
    logger.info(f"Сохранение метрик в {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path='reports/figures/confusion_matrix.png'):
    """Построение и сохранение матрицы ошибок"""
    logger.info(f"Построение матрицы ошибок в {save_path}")
    
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Создание матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    
    # Построение графика
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or 'auto',
                yticklabels=class_names or 'auto')
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    
    # Сохранение графика
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    """Главная функция для оценки модели"""
    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
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
        # Загрузка тестовых данных из S3 или локально
        test_data_path = os.path.join(config['data']['processed_path'], config['data']['test_file'])
        test_target_path = test_data_path.replace('.csv', '_target.csv')
        
        if s3_bucket:
            X_test, y_test = load_processed_data_from_s3(
                s3_bucket, 
                s3_processed_prefix + config['data']['test_file'],
                s3_processed_prefix + config['data']['test_file'].replace('.csv', '_target.csv'),
                aws_access_key_id, 
                aws_secret_access_key
            )
        else:
            X_test, y_test = load_processed_data(test_data_path, test_target_path)
        
        # Загрузка обученной модели из S3 или локально
        model_path = "models/model.pkl"
        if s3_bucket:
            model = load_model_from_s3(s3_bucket, 'models/model.pkl', 
                                       aws_access_key_id, aws_secret_access_key)
        else:
            model = load_model(model_path)
        
        # Оценка модели
        test_metrics, y_pred = evaluate_model(model, X_test, y_test)
        logger.info(f"Тестовые метрики: {test_metrics}")
        
        # Сохранение метрик
        save_metrics(test_metrics, "metrics.json")
        
        # Построение матрицы ошибок
        plot_confusion_matrix(y_test, y_pred, save_path='reports/figures/confusion_matrix.png')
        
        # Загрузка графиков и метрик в S3, если указан bucket
        if s3_bucket:
            upload_to_s3('reports/figures/confusion_matrix.png', s3_bucket, 
                         'reports/figures/confusion_matrix.png', 
                         aws_access_key_id, aws_secret_access_key)
            upload_to_s3('metrics.json', s3_bucket, 'metrics.json', 
                         aws_access_key_id, aws_secret_access_key)
        
        # Логирование в MLflow, если URI указан
        if mlflow_tracking_uri:
            # Получение активного эксперимента
            experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
            if experiment:
                # Поиск последнего запуска
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                          order_by=["start_time DESC"], 
                                          max_results=1)
                if not runs.empty:
                    run_id = runs.iloc[0]['run_id']
                    with mlflow.start_run(run_id=run_id):
                        # Логирование метрик
                        mlflow.log_metrics(test_metrics)
                        
                        # Логирование артефактов
                        mlflow.log_artifact('reports/figures/confusion_matrix.png')
                        mlflow.log_artifact('metrics.json')
        
        logger.info("Оценка модели успешно завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в оценке модели: {str(e)}")
        raise

if __name__ == "__main__":
    main()
