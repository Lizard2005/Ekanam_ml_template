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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(data_path, target_path):
    """Загрузка обработанных данных и целевых переменных"""
    logger.info(f"Загрузка обработанных данных из {data_path}")
    X = pd.read_csv(data_path)
    y = pd.read_csv(target_path)['target']
    return X, y

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

def main():
    """Главная функция для оценки модели"""
    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Загрузка тестовых данных
        test_data_path = os.path.join(config['data']['processed_path'], config['data']['test_file'])
        test_target_path = test_data_path.replace('.csv', '_target.csv')
        
        X_test, y_test = load_processed_data(test_data_path, test_target_path)
        
        # Загрузка обученной модели
        model_path = "models/model.pkl"
        model = load_model(model_path)
        
        # Оценка модели
        test_metrics, y_pred = evaluate_model(model, X_test, y_test)
        logger.info(f"Тестовые метрики: {test_metrics}")
        
        # Сохранение метрик
        save_metrics(test_metrics, "metrics.json")
        
        # Построение матрицы ошибок
        plot_confusion_matrix(y_test, y_pred, save_path='reports/figures/confusion_matrix.png')
        
        logger.info("Оценка модели успешно завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в оценке модели: {str(e)}")
        raise

if __name__ == "__main__":
    main()
