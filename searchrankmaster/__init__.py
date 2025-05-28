"""SearchRankMaster: A machine learning-powered search ranking system."""

__version__ = "0.1.0"

# Import key components for easier access
from .data import DataLoader, prepare_dataset_for_training
from .models import Ranker, available_models
from .evaluation import evaluate_model, plot_metrics

__all__ = [
    "DataLoader",
    "prepare_dataset_for_training",
    "Ranker",
    "available_models",
    "evaluate_model",
    "plot_metrics",
]
