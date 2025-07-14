install:
	pip install -r requirements.txt
	pip install -r requirements.dev.txt

test:
	pytest tests/

lint:
	ruff check src tests

format:
	ruff format src tests

run:
	python src/web_service.py

ci: install lint test
