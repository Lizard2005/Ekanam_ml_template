# Ekanam ml template
Производственный шаблон репозитория для разработки машинного обучения, поддерживающий экспериментирование, отслеживание версий данных и моделей, а также легкий переход к продакшен-развертыванию.

## 🚀 Особенности
- Версионирование данных и моделей с помощью DVC

- Автоматизация workflows через Makefile

- Современное управление зависимостями с помощью UV

- Шаблонизация проектов с помощью Cookiecutter

- Структурированная организация кода по функциональности

- Логирование экспериментов и метрик

- Конфигурационное управление через YAML-файлы

- Готовность к продакшену с Docker и API-интерфейсом

- Тестирование и линтинг для поддержания качества кода

📁 Структура проекта
```
ekanam-ml-template/
├── configs/           # Конфигурационные файлы
│   └── config.yaml   # Основной конфигурационный файл
├── data/             # Данные (управляются через DVC)
├── logs/             # Логи выполнения
├── models/           # Обученные модели (DVC)
├── notebooks/        # Jupyter ноутбуки для исследований
├── scripts/          # Скрипты для автоматизации
│   └── setup_environment.sh  # Скрипт настройки окружения
├── src/              # Исходный код проекта
│   └── data/         # Код для обработки данных
│       ├── __init__.py
│       └── make_dataset.py
├── tests/            # Тесты
│   └── test_data.py  # Тесты для модуля данных
├── cookiecutter.json # Конфигурация Cookiecutter
├── .dvcignore        # Ignore-файл для DVC
├── .env              # Переменные окружения
├── .gitignore        # Ignore-файл для Git
├── dvc.yaml          # DVC pipeline
├── Makefile          # Автоматизация задач
├── params.yaml       # Параметры для обучения моделей
├── pyproject.toml    # Конфигурация проекта и зависимости
├── setup.py          # Установка пакета
└── README.md         # Документация
```

## 🍪 Работа с Cookiecutter
Cookiecutter позволяет создавать новые проекты из шаблона с автоматической подстановкой параметров.

### Установка Cookiecutter
``` 
# Установка через pip
pip install cookiecutter

# Или установка через UV
uv pip install cookiecutter
```
### Создание нового проекта из шаблона
#### Создайте новый проект из шаблона:

```
cookiecutter https://github.com/yourusername/ekanam-ml-template
```
#### Ответьте на вопросы о вашем проекте:

```
project_name [My ML Project]: Customer Churn Prediction
repo_name [customer_churn_prediction]: 
author_name [Your Name]: Data Science Team
email [your.email@example.com]: team@company.com
description [A machine learning project]: Predict customer churn using machine learning
python_version [3.9]: 3.10
```
#### Перейдите в созданную директорию:

```
cd customer_churn_prediction
```
### Параметры Cookiecutter
При создании проекта вам будет предложено указать следующие параметры:
```
Параметр	Описание	По умолчанию
project_name	Название проекта	"My ML Project"
repo_name	Имя репозитория	Автоматически генерируется
author_name	Имя автора	"Your Name"
email	Email автора	"your.email@example.com"
description	Описание проекта	"A machine learning project"
python_version	Версия Python	"3.9"
```
### Настройка созданного проекта
После создания проекта из шаблона:

#### Настройте окружение:

```
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```
#### Активируйте виртуальное окружение:

```
source .venv/bin/activate
```
#### Настройте DVC remote storage (если нужно):

```
dvc remote add -d myremote /path/to/remote/storage
```
### Преимущества использования Cookiecutter
- Стандартизация: Все проекты команды имеют одинаковую структуру

- Экономия времени: Автоматическое создание проекта с готовой конфигурацией

- Автоматическая подстановка: Имена проектов, авторов и другие параметры подставляются автоматически

- Легкое обновление: Можно обновлять шаблон и применять изменения к новым проектам

###Для разработчиков шаблона
Если вы вносите изменения в шаблон:

#### Протестируйте создание проекта:

```
cookiecutter . --overwrite-if-exists
Проверьте сгенерированный проект:
```
#### Проверьте сгенерированный проект:
```
cd test_project
./scripts/setup_environment.sh
make test
``` 

## ⚡ Быстрый старт
### Предварительные требования
- Python 3.9+

- Git

- DVC (для управления данными и моделями)

- Make (рекомендуется)

## Настройка окружения
### Клонируйте репозиторий:
```
git clone <your-repo-url>
cd ml-project-template
```
### Запустите скрипт настройки окружения:
```
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```
### Активируйте виртуальное окружение:
```
source .venv/bin/activate
```

## 🛠 Использование
### Основные команды
```
# Установка зависимостей
make install

# Запуск тестов
make test

# Проверка кодстайла
make lint

# Подготовка данных
make data

# Обучение модели
make train

# Оценка модели
make evaluate

# Запуск API сервера
make serve

# Сборка Docker образа
make docker
```
### Работа с данными
Данные и модели управляются через DVC:
```
# Добавление файла данных под версионный контроль
dvc add data/raw/dataset.csv

# Синхронизация с удаленным хранилищем
dvc push

# Получение последней версии данных
dvc pull
```
### Запуск экспериментов
1. Настройте параметры в params.yaml

2. Запустите обучение:
```
python src/models/train.py
```
3. Отслеживайте эксперименты с помощью MLflow (автоматически запускается)

## ⚙️ Конфигурация
### Проект использует конфигурационные файлы в формате YAML:

```configs/config.yaml``` - основные настройки проекта

```params.yaml``` - параметры для обучения моделей

## 📦 Развертывание
### Docker
Соберите и запустите контейнер:
```
make docker
docker run -p 8000:8000 ml-project
```
### API сервер
Проект включает FastAPI сервер для предсказаний:
```
make serve
```
API будет доступно по адресу: http://localhost:8000/docs

### Cloud развертывание
Пример для AWS SageMaker:
```
# Упаковка модели для SageMaker
python src/models/pack_for_sagemaker.py

# Деплой (пример для AWS CLI)
aws s3 sync sagemaker_model/ s3://my-bucket/models/my-model/
```
## 🔬 Эксперименты и отслеживание
Проект настроен для работы с MLflow для отслеживания экспериментов:
```
import mlflow

# Автоматическое логирование параметров и метрик
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Ваш код обучения
    model.fit(X_train, y_train)
```
Для просмотра результатов:
```
mlflow ui
```
## 📝 Скрипты
В папке scripts/ размещаются дополнительные скрипты для автоматизации common сценариев. Основной скрипт:

- ```setup_environment.sh``` - настройка окружения с установкой UV и зависимостей

## 🤝 Вклад в проект
1. Форкните репозиторий

2. Создайте ветку для функциональности (```git checkout -b feature/amazing-feature```)

3. Закоммитьте изменения (```git commit -m 'Add amazing feature'```)

4. Запушьте ветку (```git push origin feature/amazing-feature```)

5. Откройте Pull Request