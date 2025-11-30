"""Setup configuration for LoL Esports Prediction package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="lol-esports-prediction",
    version="0.1.0",
    description="Machine Learning pipeline for predicting League of Legends esports match outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lol-esports-prediction",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    install_requires=[
        "numpy>=1.24.3,<2.0.0",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.6",
        "mlflow>=2.7.1",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "shap>=0.42.1",
        "joblib>=1.3.2",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "isort>=5.13.2",
            "pre-commit>=3.6.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
