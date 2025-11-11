"""
Data loading and processing utilities.

This module contains functions for loading raw data and performing
initial data quality checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the raw data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or corrupted
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath}")
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise ValueError(f"Failed to load data from {filepath}")


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform basic data quality checks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    logger.info("Data quality check completed")
    return quality_report


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'mode')
        
    Returns:
        DataFrame with missing values handled
    """
    if strategy == 'drop':
        df_clean = df.dropna()
        logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        logger.info("Filled missing values with mean")
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        logger.info("Filled missing values with median")
    elif strategy == 'mode':
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        logger.info("Filled missing values with mode")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save processed data to a CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False)
    logger.info(f"Saved processed data to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Data loading module loaded successfully")
