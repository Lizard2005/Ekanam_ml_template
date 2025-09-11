import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import joblib
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def load_data(data_path, target_path):
    """Загрузка обработанных данных и целевых переменных"""
    print(f"Загрузка данных из {data_path}")
    X = pd.read_csv(data_path)
    y = pd.read_csv(target_path)['target']
    return X, y

def load_model(model_path):
    """Загрузка обученной модели"""
    print(f"Загрузка модели из {model_path}")
    return joblib.load(model_path)

def plot_feature_distributions(X, save_path='reports/figures/distribution_plots.png'):
    """Построение распределений признаков"""
    print("Создание графиков распределения признаков")
    
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Определение количества признаков и создание сетки подграфиков
    n_features = min(len(X.columns), 12)  # Ограничение до 12 признаков для читаемости
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Создание подграфиков
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    # Построение распределения для каждого признака
    for i, col in enumerate(X.columns[:n_features]):
        axes[i].hist(X[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Распределение {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Частота')
        axes[i].grid(True, alpha=0.3)
    
    # Скрытие неиспользуемых подграфиков
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Графики распределения сохранены в {save_path}")

def plot_correlation_heatmap(X, save_path='reports/figures/correlation_heatmap.png'):
    """Построение тепловой карты корреляции признаков"""
    print("Создание тепловой карты корреляции")
    
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Расчет матрицы корреляции
    corr_matrix = X.corr()
    
    # Создание тепловой карты
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Маска для верхнего треугольника
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    plt.title('Тепловая карта корреляции признаков')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Тепловая карта корреляции сохранена в {save_path}")

def plot_feature_importance(model, feature_names, save_path='plots/feature_importance.png'):
    """Построение важности признаков, если доступна"""
    print("Создание графика важности признаков")
    
    # Проверка наличия атрибута важности признаков у модели
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):  # Для линейных моделей
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        print("Модель не имеет атрибутов важности признаков")
        return
    
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Создание DataFrame для построения графика
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Построение графика
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
    sns.barplot(data=feature_importance_df, 
                y='feature', 
                x='importance', 
                palette='viridis')
    plt.title('Важность признаков')
    plt.xlabel('Важность')
    plt.ylabel('Признаки')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График важности признаков сохранен в {save_path}")

def plot_roc_curve(model, X_test, y_test, save_path='reports/figures/roc_curve.png'):
    """Построение ROC кривой"""
    print("Создание ROC кривой")
    
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Проверка наличия метода predict_proba у модели
    if not hasattr(model, 'predict_proba'):
        print("Модель не поддерживает вероятностные предсказания")
        return
    
    # Получение предсказанных вероятностей
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Расчет ROC кривой
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC кривая (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Случайный классификатор')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Доля ложноположительных результатов')
    plt.ylabel('Доля истинноположительных результатов')
    plt.title('Кривая ошибок приемника (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC кривая сохранена в {save_path}")

def main():
    """Главная функция для генерации визуализаций"""
    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Загрузка обучающих данных для графиков распределения и корреляции
        train_data_path = os.path.join(config['data']['processed_path'], config['data']['train_file'])
        train_target_path = train_data_path.replace('.csv', '_target.csv')
        
        X_train, y_train = load_data(train_data_path, train_target_path)
        
        # Генерация графиков распределения
        plot_feature_distributions(X_train, 'reports/figures/distribution_plots.png')
        
        # Генерация тепловой карты корреляции
        plot_correlation_heatmap(X_train, 'reports/figures/correlation_heatmap.png')
        
        # Загрузка модели для графика важности признаков
        model_path = "models/model.pkl"
        if os.path.exists(model_path):
            model = load_model(model_path)
            
            # Генерация графика важности признаков
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = X_train.columns
            
            plot_feature_importance(model, feature_names, 'plots/feature_importance.png')
            
            # Загрузка тестовых данных для ROC кривой
            test_data_path = os.path.join(config['data']['processed_path'], config['data']['test_file'])
            test_target_path = test_data_path.replace('.csv', '_target.csv')
            
            if os.path.exists(test_data_path) and os.path.exists(test_target_path):
                X_test, y_test = load_data(test_data_path, test_target_path)
                
                # Генерация ROC кривой
                plot_roc_curve(model, X_test, y_test, 'reports/figures/roc_curve.png')
        else:
            print(f"Файл модели {model_path} не найден. Пропуск визуализаций, зависящих от модели.")
        
        print("Генерация визуализаций успешно завершена")
        
    except Exception as e:
        print(f"Ошибка в генерации визуализаций: {str(e)}")
        raise

if __name__ == "__main__":
    main()
