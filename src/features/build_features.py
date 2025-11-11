"""
Feature engineering utilities.

This module contains functions for creating and transforming features
for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class for performing feature engineering operations.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create temporal features from a date column.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Created temporal features from {date_column}")
        return df
    
    def create_aggregation_features(self, df: pd.DataFrame, 
                                   group_col: str, 
                                   agg_col: str,
                                   agg_funcs: list = ['mean', 'sum', 'std']) -> pd.DataFrame:
        """
        Create aggregation features based on grouping.
        
        Args:
            df: Input DataFrame
            group_col: Column to group by
            agg_col: Column to aggregate
            agg_funcs: List of aggregation functions
            
        Returns:
            DataFrame with aggregation features
        """
        df = df.copy()
        
        for func in agg_funcs:
            feature_name = f"{agg_col}_{func}_by_{group_col}"
            df[feature_name] = df.groupby(group_col)[agg_col].transform(func)
            
        logger.info(f"Created aggregation features for {agg_col} grouped by {group_col}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   columns: list,
                                   method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode
            method: Encoding method ('label' or 'onehot')
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        if method == 'label':
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.encoders[col].fit_transform(df[col])
                else:
                    df[f"{col}_encoded"] = self.encoders[col].transform(df[col])
                    
            logger.info(f"Label encoded {len(columns)} categorical features")
            
        elif method == 'onehot':
            df = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=True)
            logger.info(f"One-hot encoded {len(columns)} categorical features")
            
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                                columns: list,
                                method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns to scale
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if method == 'standard':
            for col in columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[f"{col}_scaled"] = self.scalers[col].fit_transform(df[[col]])
                else:
                    df[f"{col}_scaled"] = self.scalers[col].transform(df[[col]])
                    
            logger.info(f"Standard scaled {len(columns)} numerical features")
            
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   col1: str, 
                                   col2: str,
                                   operation: str = 'multiply') -> pd.DataFrame:
        """
        Create interaction features between two columns.
        
        Args:
            df: Input DataFrame
            col1: First column
            col2: Second column
            operation: Type of interaction ('multiply', 'divide', 'add', 'subtract')
            
        Returns:
            DataFrame with interaction feature
        """
        df = df.copy()
        
        if operation == 'multiply':
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        elif operation == 'divide':
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-10)
        elif operation == 'add':
            df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
        elif operation == 'subtract':
            df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
            
        logger.info(f"Created interaction feature: {col1} {operation} {col2}")
        return df
    
    def create_binned_features(self, df: pd.DataFrame, 
                              column: str,
                              bins: int = 5,
                              labels: list = None) -> pd.DataFrame:
        """
        Create binned categorical features from continuous variables.
        
        Args:
            df: Input DataFrame
            column: Column to bin
            bins: Number of bins or list of bin edges
            labels: Labels for the bins
            
        Returns:
            DataFrame with binned feature
        """
        df = df.copy()
        
        if labels is None:
            labels = [f"bin_{i}" for i in range(bins if isinstance(bins, int) else len(bins)-1)]
            
        df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
        
        logger.info(f"Created binned feature for {column}")
        return df


def calculate_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Calculate and return feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Calculated feature importance")
    return importance_df


if __name__ == "__main__":
    print("Feature engineering module loaded successfully")
