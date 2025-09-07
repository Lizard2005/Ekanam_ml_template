#!/bin/bash

# Создание виртуального окружения
python -m venv .venv

# Активация окружения
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements/dev.txt
pip install -e .

# Инициализация DVC
dvc init

echo "Environment setup complete!"