#!/bin/bash

# Проверяем, установлен ли uv
if ! command -v uv &> /dev/null; then
    echo "UV не установлен. Установка..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Создание виртуального окружения с помощью uv
uv venv

# Активация окружения
source .venv/bin/activate

# Установка зависимостей для разработки
uv pip install -e .[dev]

# Проверяем, установлен ли DVC
if ! command -v dvc &> /dev/null; then
    echo "DVC не установлен. Установка..."
    uv pip install dvc
fi

# Инициализация DVC
dvc init

echo "Настройка окружения завершена!"
echo "Активируйте виртуальное окружение: source .venv/bin/activate"
