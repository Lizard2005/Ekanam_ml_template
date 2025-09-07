.PHONY: help install test clean data train evaluate

help:
	@echo "Доступные команды:"
	@echo "  make install     Установить зависимости"
	@echo "  make test        Запустить тесты"
	@echo "  make lint        Проверить код линтером"
	@echo "  make data        Загрузить и подготовить данные"
	@echo "  make train       Обучить модель"
	@echo "  make evaluate    Оценить модель"
	@echo "  make serve       Запустить API сервер"
	@echo "  make docker      Собрать Docker образ"

install:
	pip install -r requirements/dev.txt
	pip install -e .

test:
	pytest tests/ -v

lint:
	flake8 src/
	black --check src/ tests/
	isort --check-only src/ tests/

data:
	python src/data/make_dataset.py

train:
	python src/models/train.py

evaluate:
	python src/models/evaluate.py

serve:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

docker:
	docker build -t ml-project .