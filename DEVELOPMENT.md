# Development Guide

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <your-repo-url>
cd Personal_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode with dev dependencies
pip install -e .[dev]
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code quality checks before each commit.

### 3. Configure Environment (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your preferences (optional)
nano .env
```

## Development Workflow

### Running Code Quality Checks

```bash
# Format code with Black
black src tests

# Sort imports with isort
isort src tests

# Lint code with Flake8
flake8 src tests

# Type check with mypy
mypy src --ignore-missing-imports

# Security check with Bandit
bandit -r src

# Or run all checks at once
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# View coverage in browser
open htmlcov/index.html  # macOS
# or
start htmlcov/index.html  # Windows
```

### Running the Pipeline

```bash
# 1. Preprocess data
python scripts/run_preprocessing.py

# 2. Create PCA features
python scripts/run_feature_engineering.py

# 3. Train all models
python scripts/train_all_models.py

# Or train a single model
python scripts/train_single_model.py gradient_boosting

# With MLflow tracking
python scripts/train_all_models.py --mlflow
mlflow ui
```

## Project Structure

```
Personal_Project/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/               # Feature engineering
│   │   └── engineering.py
│   ├── models/                 # Model training
│   │   ├── base.py
│   │   └── train.py
│   ├── evaluation/             # Metrics and visualization
│   │   └── metrics.py
│   └── utils/                  # Utilities
│       ├── logging_config.py   # Centralized logging
│       └── validation.py       # Data validation
├── tests/                      # Unit tests
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_validation.py
│   └── test_metrics.py
├── scripts/                    # Executable scripts
├── notebooks/                  # Jupyter notebooks
├── data/                       # Data files (gitignored)
├── models/                     # Saved models (gitignored)
├── results/                    # Results (gitignored)
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml              # Tool configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                    # Package configuration
├── README.md
├── CODE_REVIEW.md              # Improvements summary
└── DEVELOPMENT.md              # This file
```

## Code Quality Standards

### Type Hints

All functions should have type hints:

```python
def train_model(
    model_type: str,
    data: Optional[pd.DataFrame] = None,
    tune_hyperparameters: bool = True
) -> BaseModel:
    """Train a model with optional hyperparameter tuning."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities (optional).

    Returns:
        Dictionary containing all metrics.

    Example:
        >>> metrics = calculate_metrics(y_true, y_pred, y_proba)
        >>> print(metrics['accuracy'])
        0.95
    """
    ...
```

### Testing

Write tests for all new functionality:

```python
def test_load_processed_data_invalid_type():
    """Test that ValueError is raised for invalid data type."""
    with pytest.raises(ValueError) as exc_info:
        load_processed_data("invalid_type")

    assert "Invalid data_type" in str(exc_info.value)
```

## Continuous Integration

Every push to `main` or `develop` branches triggers:

1. **Code Quality Checks**
   - Black formatting verification
   - isort import sorting verification
   - Flake8 linting
   - mypy type checking
   - Bandit security scanning

2. **Testing**
   - Tests on Ubuntu and macOS
   - Python versions: 3.9, 3.10, 3.11
   - Coverage reporting to Codecov

## Tools & Configuration

### Black (Code Formatting)
- Line length: 88 characters
- Configuration: `pyproject.toml`

### isort (Import Sorting)
- Profile: black
- Configuration: `pyproject.toml`

### Flake8 (Linting)
- Max line length: 88
- Ignores: E203, W503 (Black compatibility)
- Configuration: `pyproject.toml`

### mypy (Type Checking)
- Python version: 3.9+
- Configuration: `pyproject.toml`

### pytest (Testing)
- Minimum coverage: 80%
- Configuration: `pyproject.toml`

## Common Tasks

### Add a New Model

1. Add model configuration to `src/models/train.py`:
```python
MODEL_CONFIGS = {
    'your_model': {
        'model': YourModelClass(),
        'param_grid': {...},
        'use_scaler': False
    }
}
```

2. Add tests in `tests/test_models.py`

3. Run tests: `pytest tests/test_models.py`

### Add New Features

1. Modify `src/features/engineering.py`
2. Update `src/config.py` if needed
3. Add tests in `tests/test_features.py`
4. Run quality checks: `pre-commit run --all-files`

### Update Dependencies

```bash
# Add to requirements.txt or setup.py
pip install <package>

# Update requirements files
pip freeze > requirements.txt

# Or use pip-tools for better management
pip-compile requirements.in
```

## Troubleshooting

### Pre-commit hooks failing

```bash
# Run hooks manually to see detailed errors
pre-commit run --all-files

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

### Import errors

```bash
# Reinstall in editable mode
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Test failures

```bash
# Run specific test
pytest tests/test_file.py::TestClass::test_method -v

# Run with debugging
pytest --pdb

# Update test data if schema changed
pytest --fixtures
```

## Resources

- **Black**: https://black.readthedocs.io/
- **isort**: https://pycqa.github.io/isort/
- **pytest**: https://docs.pytest.org/
- **mypy**: https://mypy.readthedocs.io/
- **pre-commit**: https://pre-commit.com/
