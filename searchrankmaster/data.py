"""Data loading and preprocessing utilities for SearchRankMaster."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from config import settings

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and processing of ranking datasets."""
    
    def __init__(self, dataset_name: str, data_dir: Optional[Path] = None):
        """Initialize the data loader.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'mslr_web10k')
            data_dir: Directory containing the dataset files
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir or settings.DATA_DIR
        self._validate_dataset_name()
    
    def _validate_dataset_name(self) -> None:
        """Validate the dataset name."""
        if self.dataset_name not in settings.DATASET_NAMES:
            raise ValueError(
                f"Invalid dataset name: {self.dataset_name}. "
                f"Must be one of: {list(settings.DATASET_NAMES.keys())}"
            )
    
    def get_split_path(self, split: str) -> Path:
        """Get the file path for a dataset split.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            
        Returns:
            Path to the dataset file
        """
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'")
            
        filename = f"{split}.parquet"
        path = self.data_dir / self.dataset_name / filename
        
        # Fall back to CSV if parquet doesn't exist
        if not path.exists():
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                return csv_path
                
        return path
    
    def load_split(
        self,
        split: str,
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        use_dask: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load a dataset split.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            columns: List of columns to load (None for all)
            nrows: Maximum number of rows to load
            use_dask: Whether to use Dask for out-of-core processing
            **kwargs: Additional arguments to pass to the reader
            
        Returns:
            Loaded dataset as a pandas or Dask DataFrame
        """
        path = self.get_split_path(split)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        # Determine the file format
        is_parquet = path.suffix.lower() == '.parquet'
        
        try:
            if use_dask:
                return self._load_with_dask(path, is_parquet, columns, nrows, **kwargs)
            else:
                return self._load_with_pandas(path, is_parquet, columns, nrows, **kwargs)
        except Exception as e:
            logger.error(f"Error loading {split} split: {str(e)}")
            raise
    
    def _load_with_pandas(
        self,
        path: Path,
        is_parquet: bool,
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data using pandas."""
        if is_parquet:
            return pd.read_parquet(path, columns=columns, **kwargs)
        else:
            return pd.read_csv(path, nrows=nrows, usecols=columns, **kwargs)
    
    def _load_with_dask(
        self,
        path: Path,
        is_parquet: bool,
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        **kwargs
    ) -> dd.DataFrame:
        """Load data using Dask for out-of-core processing."""
        if is_parquet:
            df = dd.read_parquet(path, columns=columns, **kwargs)
        else:
            df = dd.read_csv(path, usecols=columns, **kwargs)
            
        if nrows is not None:
            df = df.head(nrows, compute=False)
            
        return df
    
    def load_in_chunks(
        self,
        split: str,
        chunk_size: int = settings.CHUNK_SIZE,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """Load dataset in chunks.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            chunk_size: Number of rows per chunk
            columns: List of columns to load
            **kwargs: Additional arguments to pass to the reader
            
        Yields:
            Chunks of the dataset as pandas DataFrames
        """
        path = self.get_split_path(split)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
            
        is_parquet = path.suffix.lower() == '.parquet'
        
        try:
            if is_parquet:
                # For parquet, we can use fastparquet or pyarrow's chunking
                import pyarrow.parquet as pq
                
                # Get the schema to determine column types
                schema = pq.read_schema(path)
                
                # Create a parquet file reader
                parquet_file = pq.ParquetFile(path)
                
                # Read in chunks
                for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
                    yield batch.to_pandas()
            else:
                # For CSV, use pandas' chunking
                for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=columns, **kwargs):
                    yield chunk
        except Exception as e:
            logger.error(f"Error loading {split} in chunks: {str(e)}")
            raise

def prepare_dataset_for_training(
    df: pd.DataFrame, 
    features_to_use: Optional[List[int]] = None,
    target_col: str = 'relevance',
    query_id_col: str = 'query_id',
    return_dataframe: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], 
          Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """Prepare dataset for model training with comprehensive validation.
    
    Args:
        df: Input DataFrame containing features and target
        features_to_use: Optional list of feature indices to use (if None, use all features)
        target_col: Name of the target column
        query_id_col: Name of the query ID column
        return_dataframe: If True, return pandas objects instead of numpy arrays
        
    Returns:
        Tuple of (X, y, qids) where:
        - X: Feature matrix (numpy array or DataFrame)
        - y: Target values (numpy array or Series)
        - qids: Query IDs (numpy array or Series)
        
    Raises:
        TypeError: If input types are incorrect
        ValueError: If input data is invalid or feature indices are out of range
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
        
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Check required columns
    required_columns = {target_col, query_id_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get feature columns (columns that start with 'f' and are numeric)
    feature_cols = [col for col in df.columns 
                   if col.startswith('f') and pd.api.types.is_numeric_dtype(df[col])]
    
    if not feature_cols:
        raise ValueError("No valid feature columns found in DataFrame")
    
    # Filter features if specified
    if features_to_use is not None:
        if not isinstance(features_to_use, (list, np.ndarray)):
            raise TypeError(f"features_to_use must be a list or array, got {type(features_to_use).__name__}")
            
        if not all(isinstance(f, (int, np.integer)) for f in features_to_use):
            raise TypeError("All feature indices must be integers")
            
        if not all(0 <= f < len(feature_cols) for f in features_to_use):
            raise ValueError(f"Feature indices must be between 0 and {len(feature_cols)-1}")
            
        feature_cols = [feature_cols[i] for i in features_to_use]
    
    # Extract data
    X = df[feature_cols]
    y = df[target_col]
    qids = df[query_id_col]
    
    # Validate shapes
    if len(X) != len(y) or len(X) != len(qids):
        raise ValueError(
            f"Mismatch in number of samples: "
            f"X={len(X)}, y={len(y)}, qids={len(qids)}"
        )
    
    # Convert to numpy arrays if needed
    if not return_dataframe:
        X = X.values if hasattr(X, 'values') else X.to_numpy()
        y = y.values if hasattr(y, 'values') else y.to_numpy()
        qids = qids.values if hasattr(qids, 'values') else qids.to_numpy()
    
    return X, y, qids
