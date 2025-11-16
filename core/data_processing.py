"""
Data processing utilities for loading, cleaning, and preprocessing data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading utilities for various file formats."""

    @staticmethod
    def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Loaded CSV with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV {filepath}: {e}")
            raise

    @staticmethod
    def load_excel(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load Excel file with error handling."""
        try:
            df = pd.read_excel(filepath, **kwargs)
            logger.info(f"Loaded Excel with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load Excel {filepath}: {e}")
            raise

    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
        """Load JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON {filepath}: {e}")
            raise


class DataPreprocessor:
    """Data preprocessing utilities."""

    @staticmethod
    def handle_missing_values(df: pd.DataFrame,
                            strategy: str = 'drop',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        df = df.copy()

        if columns is None:
            columns = df.columns

        if strategy == 'drop':
            df = df.dropna(subset=columns)
        elif strategy == 'mean':
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            for col in columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else df[col])

        logger.info(f"Applied {strategy} strategy for missing values")
        return df

    @staticmethod
    def normalize_numeric_columns(df: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                method: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
        """Normalize numeric columns and return scalers."""
        df = df.copy()
        scalers = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if method == 'standard':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = (df[col] - mean_val) / std_val
                scalers[col] = {'mean': mean_val, 'std': std_val}
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
                scalers[col] = {'min': min_val, 'max': max_val}

        logger.info(f"Normalized {len(columns)} columns using {method} scaling")
        return df, scalers

    @staticmethod
    def encode_categorical_columns(df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 method: str = 'label') -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical columns."""
        df = df.copy()
        encoders = {}

        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            if method == 'label':
                unique_vals = df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[col] = df[col].map(mapping)
                encoders[col] = {'type': 'label', 'mapping': mapping}
            elif method == 'onehot':
                # This would create multiple columns, simplified for now
                pass

        logger.info(f"Encoded {len(columns)} categorical columns")
        return df, encoders


class DataValidator:
    """Data validation utilities."""

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic data quality checks."""
        quality_report = {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_summary': df.describe().to_dict()
        }

        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

        quality_report['outliers'] = outliers
        return quality_report

    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, str]) -> List[str]:
        """Validate DataFrame schema against expected types."""
        issues = []

        for col, expected_type in expected_schema.items():
            if col not in df.columns:
                issues.append(f"Missing column: {col}")
                continue

            actual_type = str(df[col].dtype)
            if expected_type.lower() not in actual_type.lower():
                issues.append(f"Type mismatch for {col}: expected {expected_type}, got {actual_type}")

        return issues


def split_data(df: pd.DataFrame,
               target_column: str,
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split

    # First split: train + val vs test
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
        stratify=y_temp if y_temp.dtype == 'object' else None
    )

    # Reconstruct DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    logger.info(f"Data split - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    return train_df, val_df, test_df, pd.concat([X_test, y_test], axis=1)