import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import yaml
import os
import boto3
from io import StringIO

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_s3(bucket_name, file_key, aws_access_key_id=None, aws_secret_access_key=None):
    """Загрузка данных из S3"""
    logger.info(f"Загрузка данных из s3://{bucket_name}/{file_key}")
    
    # Создание клиента S3
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client('s3', 
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)
    else:
        # Используем IAM роли или credentials из окружения
        s3 = boto3.client('s3')
    
    # Загрузка данных
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        data = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из S3: {str(e)}")
        raise

def load_data(file_path):
    """Загрузка данных из CSV файла (локально)"""
    logger.info(f"Загрузка данных из {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл данных не найден: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df, numeric_features, categorical_features):
    """Предобработка данных путем обработки пропущенных значений и кодирования категориальных признаков"""
    logger.info("Предобработка данных")
    
    # Обработка пропущенных значений
    for col in numeric_features:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_features:
        if col in df.columns:
            df[col].fillna('Неизвестно', inplace=True)
    
    # Кодирование категориальных переменных
    df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    return df_processed

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Разделение данных на обучающую и тестовую выборки"""
    logger.info("Разделение данных на обучающую и тестовую выборки")
    
    if target_column not in df.columns:
        raise ValueError(f"Целевой столбец '{target_column}' не найден в датафрейме")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_data(train_data, test_data, train_path, test_path):
    """Сохранение обучающей и тестовой выборок в CSV файлы"""
    logger.info(f"Сохранение обучающих данных в {train_path}")
    logger.info(f"Сохранение тестовых данных в {test_path}")
    
    # Создание директорий, если они не существуют
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    # Сохранение данных
    pd.DataFrame(train_data).to_csv(train_path, index=False)
    pd.DataFrame(test_data).to_csv(test_path, index=False)
    
    # Сохранение целевых переменных
    pd.DataFrame({'target': train_data['target']}).to_csv(train_path.replace('.csv', '_target.csv'), index=False)
    pd.DataFrame({'target': test_data['target']}).to_csv(test_path.replace('.csv', '_target.csv'), index=False)

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
    """Главная функция для подготовки набора данных"""
    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Для обратной совместимости будем использовать config для всех параметров
    params = config
    
    # Параметры S3 из переменных окружения
    s3_bucket = os.getenv('S3_BUCKET')
    s3_raw_data_key = os.getenv('S3_RAW_DATA_KEY', 'data/raw/data.csv')
    s3_processed_prefix = os.getenv('S3_PROCESSED_PREFIX', 'data/processed/')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    try:
        # Загрузка сырых данных из S3 или локально
        if s3_bucket:
            df = load_data_from_s3(s3_bucket, s3_raw_data_key, aws_access_key_id, aws_secret_access_key)
        else:
            raw_data_path = os.path.join(config['data']['raw_path'], 'data.csv')
            df = load_data(raw_data_path)
        
        # Предобработка данных
        df_processed = preprocess_data(
            df, 
            params['features']['numeric_features'], 
            params['features']['categorical_features']
        )
        
        # Разделение данных
        X_train, X_test, y_train, y_test = split_data(
            df_processed, 
            config['data']['target_column'],
            params['data']['test_size'],
            params['data']['random_state']
        )
        
        # Сохранение обработанных данных локально
        train_path = os.path.join(config['data']['processed_path'], config['data']['train_file'])
        test_path = os.path.join(config['data']['processed_path'], config['data']['test_file'])
        
        # Создание директорий, если они не существуют
        os.makedirs(config['data']['processed_path'], exist_ok=True)
        
        # Сохранение данных
        X_train.to_csv(train_path, index=False)
        X_test.to_csv(test_path, index=False)
        pd.DataFrame(y_train).to_csv(train_path.replace('.csv', '_target.csv'), index=False)
        pd.DataFrame(y_test).to_csv(test_path.replace('.csv', '_target.csv'), index=False)
        
        # Загрузка обработанных данных в S3, если указан bucket
        if s3_bucket:
            upload_to_s3(train_path, s3_bucket, s3_processed_prefix + config['data']['train_file'], 
                         aws_access_key_id, aws_secret_access_key)
            upload_to_s3(test_path, s3_bucket, s3_processed_prefix + config['data']['test_file'], 
                         aws_access_key_id, aws_secret_access_key)
            upload_to_s3(train_path.replace('.csv', '_target.csv'), s3_bucket, 
                         s3_processed_prefix + config['data']['train_file'].replace('.csv', '_target.csv'), 
                         aws_access_key_id, aws_secret_access_key)
            upload_to_s3(test_path.replace('.csv', '_target.csv'), s3_bucket, 
                         s3_processed_prefix + config['data']['test_file'].replace('.csv', '_target.csv'), 
                         aws_access_key_id, aws_secret_access_key)
        
        logger.info("Подготовка данных успешно завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в подготовке данных: {str(e)}")
        raise

if __name__ == "__main__":
    main()
