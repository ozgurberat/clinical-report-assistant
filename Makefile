.PHONY: install download preprocess test lint

install:
	pip install -e ".[dev]"

download:
	python -m src.data.download_openi --out data/raw

preprocess:
	python -m src.data.preprocess --raw data/raw/reports --out data/processed

test:
	pytest tests/ -v

lint:
	ruff check src tests
