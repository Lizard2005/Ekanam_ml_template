UV = uv

.PHONY: help install install-dev install-prod test lint format data train evaluate serve docker clean

help:
	@echo "Доступные команды:"
	@echo "  make install     Установить базовые зависимости"
	@echo "  make install-dev Установить зависимости для разработки"
	@echo "  make install-prod Установить зависимости для продакшена"
	@echo "  make test        Запустить тесты"
	@echo "  make lint        Проверить код линтером"
	@echo "  make format      Форматировать код"
	@echo "  make data        Загрузить и подготовить данные"
	@echo "  make train       Обучить модель"
	@echo "  make evaluate    Оценить модель"
	@echo "  make serve       Запустить API сервер"
	@echo "  make docker      Собрать Docker образ"
	@echo "  make clean       Очистить временные файлы"

install:
	PYTHONPATH=. $(UV) pip install -e .

install-dev:
	PYTHONPATH=. $(UV) pip install -e .[dev]

install-prod:
	PYTHONPATH=. $(UV) pip install -e .[production]

test:
	PYTHONPATH=. $(UV) run pytest tests/ -v --cov=src

lint:
	PYTHONPATH=. $(UV) run flake8 src/ tests/
	PYTHONPATH=. $(UV) run black --check src/ tests/
	PYTHONPATH=. $(UV) run isort --check-only src/ tests/

format:
	PYTHONPATH=. $(UV) run black src/ tests/
	PYTHONPATH=. $(UV) run isort src/ tests/

data:
	PYTHONPATH=. $(UV) run python src/data/make_dataset.py

train:
	PYTHONPATH=. $(UV) run python src/models/train.py

evaluate:
	PYTHONPATH=. $(UV) run python src/models/evaluate.py

serve:
	PYTHONPATH=. $(UV) run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

docker:
	docker build -t ml-project .

clean:
	find . -name "*.pyc" -exec rm -f {} \;
	find . -name "__pycache__" -exec rm -rf {} \;
	find . -name ".pytest_cache" -exec rm -rf {} \;
	find . -name ".coverage" -exec rm -f {} \;