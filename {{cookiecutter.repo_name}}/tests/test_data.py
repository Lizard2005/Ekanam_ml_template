import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import yaml
from src.data.make_dataset import load_data, preprocess_data, split_data

@pytest.fixture
def sample_data():
    """Создание примера данных для тестирования"""
    data = {
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
        'target': [0, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def config():
    """Пример конфигурации для тестирования"""
    return {
        'features': {
            'numeric_features': ['age', 'income'],
            'categorical_features': ['gender', 'education']
        },
        'data': {
            'test_size': 0.2,
            'random_state': 42
        }
    }

def test_load_data(sample_data):
    """Тест загрузки данных из CSV файла"""
    # Создание временного CSV файла
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_filename = f.name
    
    try:
        # Тест загрузки данных
        loaded_data = load_data(temp_filename)
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == sample_data.shape
        assert list(loaded_data.columns) == list(sample_data.columns)
    finally:
        # Очистка временного файла
        os.unlink(temp_filename)
    
    # Тест обработки ошибок для несуществующего файла
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_preprocess_data(sample_data, config):
    """Тест предобработки данных"""
    # Тест нормальной предобработки
    processed_data = preprocess_data(
        sample_data, 
        config['features']['numeric_features'], 
        config['features']['categorical_features']
    )
    
    assert isinstance(processed_data, pd.DataFrame)
    # Проверка, что категориальные переменные закодированы
    assert 'gender_M' in processed_data.columns
    assert 'education_Master' in processed_data.columns
    # Проверка, что оригинальные категориальные столбцы удалены
    assert 'gender' not in processed_data.columns
    assert 'education' not in processed_data.columns
    
    # Тест обработки пропущенных значений
    sample_data_with_nan = sample_data.copy()
    sample_data_with_nan.loc[0, 'age'] = np.nan
    sample_data_with_nan.loc[1, 'gender'] = np.nan
    
    processed_data_with_nan = preprocess_data(
        sample_data_with_nan, 
        config['features']['numeric_features'], 
        config['features']['categorical_features']
    )
    
    # Проверка, что пропущенные значения обработаны
    assert not processed_data_with_nan.isnull().any().any()

def test_split_data(sample_data, config):
    """Тест разделения данных на обучающую и тестовую выборки"""
    # Тест нормального разделения
    X_train, X_test, y_train, y_test = split_data(
        sample_data, 
        'target',
        config['data']['test_size'],
        config['data']['random_state']
    )
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    
    # Проверка размеров
    total_samples = len(sample_data)
    test_samples = int(total_samples * config['data']['test_size'])
    train_samples = total_samples - test_samples
    
    assert len(X_train) == train_samples
    assert len(X_test) == test_samples
    assert len(y_train) == train_samples
    assert len(y_test) == test_samples
    
    # Тест обработки ошибок для несуществующего целевого столбца
    with pytest.raises(ValueError):
        split_data(sample_data, 'non_existent_column')

if __name__ == "__main__":
    pytest.main([__file__])
