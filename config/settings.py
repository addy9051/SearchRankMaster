"""Configuration settings for the SearchRankMaster application."""
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings and configurations."""
    
    # Application settings
    APP_NAME: str = "SearchRankMaster"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Dataset settings
    DATASET_NAMES: Dict[str, str] = {
        "mslr_web10k": "MSLR-WEB10K",
        "mslr_web30k": "MSLR-WEB30K"
    }
    
    # Feature configuration
    NUM_FEATURES: int = 136
    FEATURE_GROUPS: Dict[str, List[int]] = {
        # Document Field Features (1-125)
        "body": list(range(1, 26)),
        "anchor": list(range(26, 51)),
        "title": list(range(51, 76)),
        "url": list(range(76, 101)),
        "whole_doc": list(range(101, 126)),
        
        # Miscellaneous Features (126-136)
        "url_structure": [126, 127],
        "link": [128, 129],
        "rank": [130, 131],
        "quality": [132, 133],
        "click": [134, 135, 136]
    }
    
    # Model settings
    DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
        "Linear Regression (Ridge)": {
            "alpha": 1.0,
            "max_iter": 1000
        },
        "Random Forest Regressor": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        },
        # Add more default parameters as needed
    }
    
    # Evaluation settings
    EVAL_METRICS: List[str] = [
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
        "map@10", "mrr", "precision@10", "recall@10"
    ]
    
    # Data loading settings
    CHUNK_SIZE: int = 100000  # For chunked processing
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
    
    @validator('DATA_DIR', 'MODELS_DIR', pre=True)
    def create_dirs(cls, v):
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

# Initialize settings
settings = Settings()

# Create required directories
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
