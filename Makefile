.PHONY: install test clean lint format data train evaluate

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	pytest tests/ -v --cov=src

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/

# Lint code
lint:
	flake8 src/ tests/
	black --check src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Download and preprocess data
data:
	python scripts/download_data.py
	python scripts/preprocess_pipeline.py

# Train models
train:
	python scripts/train_models.py --config configs/models.yaml

# Evaluate models
evaluate:
	python scripts/evaluate_models.py --output reports/

# Full pipeline
all: data train evaluate