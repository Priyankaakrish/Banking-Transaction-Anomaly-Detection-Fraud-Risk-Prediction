DATA ?= data/sample_transactions.csv
NROWS ?=

.PHONY: help setup sample train backtest explain serve test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:   ## Install dependencies
	pip install -r requirements.txt

sample:  ## Generate synthetic data with the real schema (no Kaggle download)
	python scripts/make_sample.py data/sample_transactions.csv

train:   ## Train all four models and run the ablation
	python -m src.train --data $(DATA) --out artifacts --reports reports $(if $(NROWS),--nrows $(NROWS),)

backtest: ## Rolling-origin folds with variance
	python -m src.backtest --data $(DATA) --folds 4 $(if $(NROWS),--nrows $(NROWS),)

explain: ## SHAP global drivers and per-transaction reasons
	python -m src.explain --artifacts artifacts --data $(DATA) $(if $(NROWS),--nrows $(NROWS),)

serve:   ## Run the scoring API locally
	uvicorn serve.api:app --reload --port 8000

test:    ## Run the test suite
	python -m pytest -q tests/

smoke: sample train test  ## Everything end to end on generated data

clean:
	rm -rf artifacts/*.joblib reports/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
