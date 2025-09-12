import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from unittest.mock import patch, MagicMock
from src.models.train import get_model, train_model, evaluate_model, save_model, save_metrics
from src.models.evaluate import load_model, plot_confusion_matrix

@pytest.fixture
def sample_data():
    """Создание примера данных для тестирования"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

@pytest.fixture
def trained_model(sample_data):
    """Создание обученной модели для тестирования"""
    X, y = sample_data
    model = get_model('random_forest', {'n_estimators': 10, 'random_state': 42})
    return train_model(model, X, y)

def test_get_model():
    """Тест инициализации модели"""
    # Тест случайного леса
    rf_model = get_model('random_forest', {'n_estimators': 10, 'random_state': 42})
    assert rf_model.__class__.__name__ == 'RandomForestClassifier'
    
    # Тест логистической регрессии
    lr_model = get_model('logistic_regression', {'random_state': 42})
    assert lr_model.__class__.__name__ == 'LogisticRegression'
    
    # Тест неподдерживаемой модели
    with pytest.raises(ValueError):
        get_model('unsupported_model', {})

def test_train_model(sample_data):
    """Тест обучения модели"""
    X, y = sample_data
    model = get_model('random_forest', {'n_estimators': 10, 'random_state': 42})
    
    # Обучение модели
    trained_model = train_model(model, X, y)
    
    # Проверка, что модель обучена
    assert hasattr(trained_model, 'predict')
    assert hasattr(trained_model, 'feature_importances_')

def test_evaluate_model(sample_data, trained_model):
    """Тест оценки модели"""
    X, y = sample_data
    
    # Оценка модели
    metrics = evaluate_model(trained_model, X, y)
    
    # Проверка структуры метрик
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    # Проверка, что значения метрик находятся в допустимом диапазоне
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1

def test_save_model(trained_model):
    """Тест сохранения модели на диск"""
    # Создание временного файла
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model_path = f.name
    
    try:
        # Сохранение модели
        save_model(trained_model, model_path)
        
        # Проверка, что файл существует
        assert os.path.exists(model_path)
        
        # Проверка, что модель может быть загружена
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, 'predict')
    finally:
        # Очистка
        if os.path.exists(model_path):
            os.unlink(model_path)

def test_save_metrics():
    """Тест сохранения метрик в JSON файл"""
    metrics = {
        'accuracy': 0.95,
        'precision': 0.93,
        'recall': 0.92,
        'f1_score': 0.925
    }
    
    # Создание временного файла
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        metrics_path = f.name
    
    try:
        # Сохранение метрик
        save_metrics(metrics, metrics_path)
        
        # Проверка, что файл существует
        assert os.path.exists(metrics_path)
        
        # Проверка, что метрики могут быть загружены
        import json
        with open(metrics_path, 'r') as f:
            loaded_metrics = json.load(f)
        
        assert loaded_metrics == metrics
    finally:
        # Очистка
        if os.path.exists(metrics_path):
            os.unlink(metrics_path)

def test_load_model(trained_model):
    """Тест загрузки модели с диска"""
    # Создание временного файла
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model_path = f.name
    
    try:
        # Сохранение модели
        joblib.dump(trained_model, model_path)
        
        # Загрузка модели
        loaded_model = load_model(model_path)
        
        # Проверка, что модель загружена правильно
        assert hasattr(loaded_model, 'predict')
        assert hasattr(loaded_model, 'feature_importances_')
        
        # Тест обработки ошибок для несуществующего файла
        with pytest.raises(FileNotFoundError):
            load_model("non_existent_file.pkl")
    finally:
        # Очистка
        if os.path.exists(model_path):
            os.unlink(model_path)

@patch('matplotlib.pyplot.savefig')
def test_plot_confusion_matrix(mock_savefig):
    """Тест построения матрицы ошибок"""
    # Создание примера данных
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    
    # Тест построения графика
    plot_confusion_matrix(y_true, y_pred)
    
    # Проверка, что savefig был вызван
    mock_savefig.assert_called_once()
    
    # Тест с пользовательским путем сохранения
    custom_path = 'custom/path/confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, save_path=custom_path)
    
    # Проверка, что savefig был вызван с пользовательским путем
    mock_savefig.assert_called_with(custom_path)

if __name__ == "__main__":
    pytest.main([__file__])
