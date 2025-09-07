import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def load_data(file_path):
    """Загрузка данных из CSV файла"""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Предобработка данных"""
    logger.info("Preprocessing data")
    # Здесь добавьте вашу логику предобработки
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Разделение данных на train и test"""
    logger.info("Splitting data into train and test sets")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)