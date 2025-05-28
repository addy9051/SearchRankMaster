"""Data utility functions for SearchRankMaster."""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any, Iterator

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from config.settings import settings

# Configure logging
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

# Example usage:
if __name__ == "__main__":
    # Example of using the DataLoader
    loader = DataLoader("mslr_web10k")
    
    # Load a small sample
    df_sample = loader.load_split("train", nrows=1000)
    print(f"Loaded {len(df_sample)} rows")
    
    # Process in chunks
    total_rows = 0
    for chunk in loader.load_in_chunks("train", chunk_size=10000):
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows...")
    
    print(f"Total rows processed: {total_rows}")
