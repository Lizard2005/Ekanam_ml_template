import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import yaml
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Загрузка данных из CSV файла"""
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

def main():
    """Главная функция для подготовки набора данных"""
    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Для обратной совместимости будем использовать config для всех параметров
    params = config
    
    try:
        # Загрузка сырых данных
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
        
        # Сохранение обработанных данных
        train_path = os.path.join(config['data']['processed_path'], config['data']['train_file'])
        test_path = os.path.join(config['data']['processed_path'], config['data']['test_file'])
        
        save_data((X_train, X_test, y_train, y_test), None, train_path, test_path)
        
        logger.info("Подготовка данных успешно завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в подготовке данных: {str(e)}")
        raise

if __name__ == "__main__":
    main()
