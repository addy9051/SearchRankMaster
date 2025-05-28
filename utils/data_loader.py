"""Data loading and preparation utilities for SearchRankMaster."""
import os
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from config import settings
from utils.data_utils import DataLoader

# Feature groups in MSLR datasets
fields = ["body", "anchor", "title", "url", "whole_doc"]
metrics = [
    "cov_qt_num",            # features 1–5
    "cov_qt_ratio",          # features 6–10
    "stream_length",         # features 11–15
    "idf",                   # features 16–20
    "tf_sum",                # features 21–25
    "tf_min",                # features 26–30
    "tf_max",                # features 31–35
    "tf_mean",               # features 36–40
    "tf_var",                # features 41–45
    "norm_tf_sum",           # features 46–50
    "norm_tf_min",           # features 51–55
    "norm_tf_max",           # features 56–60
    "norm_tf_mean",          # features 61–65
    "norm_tf_var",           # features 66–70
    "tfidf_sum",             # features 71–75
    "tfidf_min",             # features 76–80
    "tfidf_max",             # features 81–85
    "tfidf_mean",            # features 86–90
    "tfidf_var",             # features 91–95
    "bool_model",            # features 96–100
    "vec_space",             # features 101–105
    "bm25",                  # features 106–110
    "lmir_abs",              # features 111–115
    "lmir_dir",              # features 116–120
    "lmir_jm",               # features 121–125
]

# Build field groups
field_groups = {}
for i, field in enumerate(fields):
    # take every 5th feature starting at i (0-based):
    ids = [i + 1 + 5 * m for m in range(len(metrics))]
    field_groups[field] = ids

# Misc feature groups
misc_groups = {
    "url_structure": [126, 127],
    "link": [128, 129],
    "rank": [130, 131],
    "quality": [132, 133],
    "click": [134, 135, 136],
}

# Combine all groups
FEATURE_GROUPS = {**field_groups, **misc_groups}

# Set up logger
import logging
logger = logging.getLogger(__name__)

def check_dataset_availability(dataset_name: str = "mslr_web10k") -> bool:
    """Check if required dataset files are available.
    
    Args:
        dataset_name: Name of the dataset to check
        
    Returns:
        bool: True if all required files exist, False otherwise
    """
    try:
        loader = DataLoader(dataset_name)
        splits = ['train', 'validation', 'test']
        missing_files = []
        
        for split in splits:
            try:
                path = loader.get_split_path(split)
                if not path.exists():
                    missing_files.append(f"{split} ({path.name})")
            except Exception as e:
                logger.warning(f"Error checking {split} split: {e}")
                missing_files.append(f"{split} (error: {str(e)})")
        
        if missing_files:
            st.error("❌ Missing or inaccessible dataset files:")
            for file in missing_files:
                st.error(f"- {file}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error checking dataset availability: {e}", exc_info=True)
        st.error(f"❌ Error checking dataset availability: {str(e)}")
        return False

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_dataset_info(dataset_name: str = "mslr_web10k") -> Optional[Dict[str, Any]]:
    """Load dataset information and statistics.
    
    Args:
        dataset_name: Name of the dataset to load info for
        
    Returns:
        Dictionary containing dataset information, or None if an error occurs
    """
    try:
        loader = DataLoader(dataset_name)
        
        # Try to load a small sample
        try:
            df = loader.load_split("train", nrows=1000)
            
            # Get total number of queries (approximate for large datasets)
            try:
                # Handle both 'qid' and 'query_id' column names
                query_col = 'qid' if 'qid' in df.columns else 'query_id'
                total_queries = df[query_col].nunique()
                
                # For large datasets, we'll estimate the total
                if len(df) >= 1000:
                    total_queries = int(total_queries * (len(df) / 1000))
                    
                # Convert to string for display
                total_queries = f"{total_queries:,}"
                    
            except Exception as e:
                logger.warning(f"Could not calculate total queries: {e}")
                total_queries = "Unknown (error calculating)"
            
            # Get feature descriptions from settings
            feature_descriptions = {
                'body': 'Features extracted from document body text (f1-f25)',
                'anchor': 'Features extracted from anchor text (f26-f50)',
                'title': 'Features extracted from document title (f51-f75)',
                'url': 'Features extracted from URL (f76-f100)',
                'whole_doc': 'Features extracted from entire document (f101-f125)',
                'url_structure': 'URL structure features (f126-f127)',
                'link': 'Inlink and outlink features (f128-f129)',
                'rank': 'PageRank and SiteRank features (f130-f131)',
                'quality': 'Quality score features (f132-f133)',
                'click': 'Click-based features (f134-f136)'
            }
            
            return {
                'name': settings.DATASET_NAMES.get(dataset_name, dataset_name),
                'description': f'Microsoft Learning to Rank Dataset ({dataset_name.upper()})',
                'num_queries': total_queries,
                'num_documents': len(df) if len(df) < 1000 else f"~{len(df):,}+",
                'features': {
                    'qid': 'Query ID',
                    'label': 'Relevance judgment (0-4)',
                    'feature_groups': feature_descriptions,
                    'total_features': settings.NUM_FEATURES
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}", exc_info=True)
            st.error(f"Error loading sample data: {str(e)}")
            
            # Return basic info even if we can't load the data
            return {
                'name': settings.DATASET_NAMES.get(dataset_name, dataset_name),
                'description': f'Microsoft Learning to Rank Dataset ({dataset_name.upper()})',
                'error': str(e),
                'features': {
                    'total_features': settings.NUM_FEATURES
                }
            }
            
    except Exception as e:
        logger.error(f"Error in load_dataset_info: {e}", exc_info=True)
        st.error(f"Error loading dataset info: {str(e)}")
        return None

@st.cache_data
def load_dataset(dataset_name, split, max_samples=None, use_dask=False, **kwargs):
    """
    Load a specific split of the dataset from local CSV files with enhanced error handling.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mslr_web10k')
        split: Dataset split ('train', 'validation', or 'test')
        max_samples: Optional maximum number of samples to load (for preview purposes)
        use_dask: Whether to use Dask for parallel loading (for large datasets)
        **kwargs: Additional arguments to pass to the underlying data loader
        
    Returns:
        Pandas DataFrame containing the loaded data, or None if an error occurs
        
    Raises:
        ValueError: If invalid dataset name or split is provided
        FileNotFoundError: If dataset file is not found
        pd.errors.EmptyDataError: If dataset file is empty
    """
    try:
        # Map split to filename for the small dataset files
        filename = {
            'train': 'small_train.csv',
            'validation': 'small_valid.csv',
            'test': 'small_test.csv'
        }.get(split)
        
        if not filename or not os.path.exists(filename):
            # Fall back to DataLoader if small files not found
            loader = DataLoader(dataset_name)
            file_path = loader.get_split_path(split)
            logger.info(f"Loading dataset from: {file_path}")
            
            if use_dask:
                import dask.dataframe as dd
                df = dd.read_csv(file_path, **kwargs).compute()
            else:
                df = pd.read_csv(file_path, **kwargs)
        else:
            logger.info(f"Loading small dataset from: {filename}")
            df = pd.read_csv(filename, **kwargs)
            
        # Limit samples if specified
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)
            
        if df is None or len(df) == 0:
            raise ValueError(f"No data loaded for {dataset_name} {split} split")
            
        # Debug: Print original columns before renaming
        logger.info(f"Original columns: {df.columns.tolist()}")
        
        # Standardize column names - handle both formats
        if 'qid' in df.columns and 'label' in df.columns:
            # Original format: qid and label
            df = df.rename(columns={'qid': 'query_id', 'label': 'relevance'})
            
            # Add feature columns if they don't exist (for small_*.csv files)
            # if not any(col.startswith('f') for col in df.columns):
            #     # The first two columns are relevance and query_id, the rest are features
            #     num_features = len(df.columns) - 2
            #     feature_columns = [f'f{i}' for i in range(1, num_features + 1)]  # Changed to 1-based indexing
            #     df.columns = ['relevance', 'query_id'] + feature_columns
            #     logger.info(f"Renamed feature columns to: {feature_columns[:5]}...")
        # elif 'query_id' in df.columns and 'relevance' in df.columns:
        #     # Already in the expected format, but check if we need to rename feature columns
        #     if any(col.startswith('feature_') for col in df.columns):
        #         # Rename feature_* columns to f1, f2, etc.
        #         feature_cols = [col for col in df.columns if col.startswith('feature_')]
        #         feature_map = {col: f'f{int(col.split("_")[1])}' for col in feature_cols}
        #         df = df.rename(columns=feature_map)
        #         logger.info(f"Renamed feature columns using mapping: {dict(list(feature_map.items())[:5])}...")
        else:
            raise ValueError("Dataset must contain either ('qid' and 'label') or ('query_id' and 'relevance') columns")
            
        # Ensure required columns exist after standardization
        required_columns = {'query_id', 'relevance'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns after standardization: {missing_columns}")
        
        # Convert all feature columns to numeric, coercing errors to NaN
        for col in df.columns:
            if col.startswith('f'):
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {e}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name} {split} split: {e}", exc_info=True)
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_data
def get_dataset_row_count(dataset_name, split):
    """Get the total number of rows in a dataset split.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mslr_web10k')
        split: Dataset split ('train', 'validation', or 'test')
        
    Returns:
        Total number of rows in the dataset split
    """
    try:
        # Map split to filename for the small dataset files
        filename = {
            'train': 'small_train.csv',
            'validation': 'small_valid.csv',
            'test': 'small_test.csv'
        }.get(split)
        
        if filename and os.path.exists(filename):
            # For small files, we can use pandas to get the row count quickly
            return sum(1 for _ in open(filename, 'r', encoding='utf-8')) - 1  # Subtract 1 for header
        else:
            # For larger files, use the DataLoader
            loader = DataLoader(dataset_name)
            file_path = loader.get_split_path(split)
            if file_path.exists():
                # Efficiently count rows without loading the entire file
                return sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
            return None
    except Exception as e:
        logger.error(f"Error getting row count for {dataset_name} {split}: {str(e)}")
        return None

@st.cache_data
def get_dataset_stats(dataset_name, split, sample_size):
    """Get basic statistics about the dataset.
    
    Args:
        dataset_name: Either 'mslr_web10k' or 'mslr_web30k'
        split: 'train', 'validation', or 'test'
        
    Returns:
        Dictionary with dataset statistics
    """
    try:
        # Get total rows in the dataset
        total_rows = get_dataset_row_count(dataset_name, split)
        
        # Load a sample of the dataset
        df_sample = load_dataset(dataset_name, split, max_samples=sample_size)
        if df_sample is None:
            return None
            
        # Get dataset info for total size
        info = load_dataset_info(dataset_name)
        if info is None:
            return None
            
        # Handle both 'qid' and 'query_id' column names
        query_col = 'qid' if 'qid' in df_sample.columns else 'query_id'
        
        # Calculate statistics
        unique_queries = df_sample[query_col].nunique()
        docs_per_query = df_sample.groupby(query_col).size().mean()
        
        # Use 'relevance' column instead of 'label'
        relevance_col = 'relevance' if 'relevance' in df_sample.columns else 'label'
        if relevance_col not in df_sample.columns:
            raise ValueError(f"Could not find relevance/label column in the dataset")
            
        relevance_dist = df_sample[relevance_col].value_counts().to_dict()
        relevance_percentage = {k: (v / len(df_sample)) * 100 for k, v in relevance_dist.items()}
        
        return {
            'total_records': info.get('num_queries', 'N/A'),
            'total_rows': total_rows,  # Add total rows to the stats
            'unique_queries_sample': unique_queries,
            'docs_per_query_avg': docs_per_query,
            'relevance_distribution': relevance_dist,
            'relevance_percentage': relevance_percentage
        }
    except Exception as e:
        st.error(f"Error calculating dataset statistics: {str(e)}")
        return None

def get_feature_names():
    """Get the names of all 136 features in the MSLR datasets.
    
    Returns:
        List of feature names
    """
    return [f'Feature {i+1}' for i in range(136)]

def get_feature_group(feature_index):
    """Get the feature group name for a given feature index.
    
    Args:
        feature_index: Index of the feature (1-136)
        
    Returns:
        Name of the feature group
    """
    # Convert to 1-based index if needed
    if isinstance(feature_index, str) and feature_index.startswith('f'):
        try:
            feature_index = int(feature_index[1:]) + 1  # Convert f0 -> 1, f1 -> 2, etc.
        except ValueError:
            return "Unknown Group"
    
    # Find the group containing this feature index
    for group_name, indices in FEATURE_GROUPS.items():
        if feature_index in indices:
            return group_name
    
    return "Unknown Group"

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
                   if col.startswith('feature_') and pd.api.types.is_numeric_dtype(df[col])]
    
    if not feature_cols:
        raise ValueError("No valid feature columns found in DataFrame")
    
    # Filter features if specified
    if features_to_use is not None:
        if not isinstance(features_to_use, (list, np.ndarray)):
            raise TypeError(f"features_to_use must be a list or array, got {type(features_to_use).__name__}")
            
        if not all(isinstance(f, (int, np.integer)) for f in features_to_use):
            raise TypeError("All feature indices must be integers")
            
        # Convert 1-based feature indices to 0-based and validate
        zero_based_features = [f - 1 for f in features_to_use]
        if not all(0 <= f < len(feature_cols) for f in zero_based_features):
            raise ValueError(f"Feature indices must be between 1 and {len(feature_cols)}")
            
        feature_cols = [feature_cols[i] for i in zero_based_features]
    
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
