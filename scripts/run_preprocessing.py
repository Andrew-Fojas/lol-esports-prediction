"""Run the complete data preprocessing pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_FILE
from src.data.preprocessor import preprocess_pipeline

if __name__ == "__main__":
    print("Starting data preprocessing pipeline...")
    preprocess_pipeline(RAW_DATA_FILE, save_intermediate=True)
    print("\nPreprocessing complete!")
