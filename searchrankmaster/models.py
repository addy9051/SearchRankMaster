"""Model definitions and training utilities for SearchRankMaster."""

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer

from .data import DataLoader, prepare_dataset_for_training
from .evaluation import ndcg_score, map_score, mrr_score

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelType = Type[BaseEstimator]

class BaseRanker(ABC, BaseEstimator, RegressorMixin):
    """Abstract base class for ranking models."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.feature_importances_ = None
        self._is_fitted = False
        
    @abstractmethod
    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        qids: Optional[ArrayLike] = None,
        **kwargs
    ) -> 'BaseRanker':
        """Fit the model to the training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            qids: Query IDs of shape (n_samples,)
            **kwargs: Additional arguments to pass to the underlying model
            
        Returns:
            self: Returns the instance itself
        """
        pass
    
    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict relevance scores for the given data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted relevance scores of shape (n_samples,)
        """
        pass
    
    def score(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        qids: Optional[ArrayLike] = None,
        metric: str = 'ndcg',
        k: int = 10,
        **kwargs
    ) -> float:
        """Score the model on the given data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: True relevance scores of shape (n_samples,)
            qids: Query IDs of shape (n_samples,)
            metric: Evaluation metric ('ndcg', 'map', or 'mrr')
            k: Truncation level for the metric
            **kwargs: Additional arguments to pass to the metric function
            
        Returns:
            Metric score (higher is better)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")
            
        y_pred = self.predict(X)
        
        if metric == 'ndcg':
            return ndcg_score(y, y_pred, qids=qids, k=k, **kwargs)
        elif metric == 'map':
            return map_score(y, y_pred, qids=qids, k=k, **kwargs)
        elif metric == 'mrr':
            return mrr_score(y, y_pred, qids=qids, k=k, **kwargs)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances.
        
        Returns:
            Feature importances as a numpy array
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")
            
        if self.feature_importances_ is not None:
            return self.feature_importances_
            
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
            
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_.flatten())
            
        logger.warning("Could not determine feature importances")
        return np.zeros(self.n_features_in_)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model to
        """
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseRanker':
        """Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        import joblib
        return joblib.load(path)

class LightGBMRanker(BaseRanker):
    """LightGBM-based ranking model."""
    
    def __init__(
        self,
        objective: str = 'lambdarank',
        metric: str = 'ndcg',
        boosting_type: str = 'gbdt',
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        **kwargs
    ):
        """Initialize the LightGBM ranker.
        
        Args:
            objective: Objective function ('lambdarank' or 'rank_xendcg')
            metric: Evaluation metric ('ndcg', 'map', 'mrr')
            boosting_type: Boosting type ('gbdt', 'dart', 'goss')
            num_leaves: Maximum number of leaves in one tree
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            **kwargs: Additional arguments to pass to lightgbm.LGBMRanker
        """
        super().__init__()
        self.objective = objective
        self.metric = metric
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        
    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        qids: Optional[ArrayLike] = None,
        **kwargs
    ) -> 'LightGBMRanker':
        """Fit the LightGBM ranker.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            qids: Query IDs of shape (n_samples,)
            **kwargs: Additional arguments to pass to lightgbm.LGBMRanker.fit()
            
        Returns:
            self: Returns the instance itself
        """
        import lightgbm as lgb
        
        if qids is None:
            raise ValueError("qids must be provided for LightGBM ranking")
            
        # Create dataset
        train_data = lgb.Dataset(
            X, 
            label=y,
            group=np.bincount(qids.astype(int))[1:],  # Group by query
            free_raw_data=False
        )
        
        # Set up parameters
        params = {
            'objective': self.objective,
            'metric': self.metric,
            'boosting_type': self.boosting_type,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            **self.kwargs
        }
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            **kwargs
        )
        
        self._is_fitted = True
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = self.model.feature_importance(importance_type='gain')
        
        return self
    
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict relevance scores.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted relevance scores of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")
            
        return self.model.predict(X, num_iteration=self.model.best_iteration or 0)

class XGBoostRanker(BaseRanker):
    """XGBoost-based ranking model."""
    
    def __init__(
        self,
        objective: str = 'rank:ndcg',
        eval_metric: str = 'ndcg',
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        **kwargs
    ):
        """Initialize the XGBoost ranker.
        
        Args:
            objective: Objective function ('rank:ndcg', 'rank:map', 'rank:pairwise')
            eval_metric: Evaluation metric ('ndcg', 'map', 'ndcg@n', 'map@n')
            max_depth: Maximum depth of a tree
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            **kwargs: Additional arguments to pass to xgboost.XGBRanker
        """
        super().__init__()
        self.objective = objective
        self.eval_metric = eval_metric
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kwargs = kwargs
    
    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        qids: Optional[ArrayLike] = None,
        **kwargs
    ) -> 'XGBoostRanker':
        """Fit the XGBoost ranker.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            qids: Query IDs of shape (n_samples,)
            **kwargs: Additional arguments to pass to xgboost.XGBRanker.fit()
            
        Returns:
            self: Returns the instance itself
        """
        import xgboost as xgb
        
        if qids is None:
            raise ValueError("qids must be provided for XGBoost ranking")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y, qid=qids)
        
        # Set up parameters
        params = {
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'max_depth': self.max_depth,
            'eta': self.learning_rate,
            **self.kwargs
        }
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            **kwargs
        )
        
        self._is_fitted = True
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = self.model.get_score(importance_type='weight')
        
        return self
    
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict relevance scores.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted relevance scores of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")
            
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

class LinearRanker(BaseRanker):
    """Linear ranking model using Ridge regression."""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        """Initialize the linear ranker.
        
        Args:
            alpha: Regularization strength (higher means more regularization)
            **kwargs: Additional arguments to pass to Ridge
        """
        super().__init__()
        self.alpha = alpha
        self.kwargs = kwargs
    
    def fit(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        qids: Optional[ArrayLike] = None,
        **kwargs
    ) -> 'LinearRanker':
        """Fit the linear ranker.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            qids: Ignored, kept for API compatibility
            **kwargs: Additional arguments to pass to Ridge.fit()
            
        Returns:
            self: Returns the instance itself
        """
        from sklearn.linear_model import Ridge
        
        self.model = Ridge(alpha=self.alpha, **self.kwargs)
        self.model.fit(X, y, **kwargs)
        
        self._is_fitted = True
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.abs(self.model.coef_)
        
        return self
    
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict relevance scores.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted relevance scores of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")
            
        return self.model.predict(X)

def available_models() -> Dict[str, Type[BaseRanker]]:
    """Get a dictionary of available ranking models.
    
    Returns:
        Dictionary mapping model names to model classes
    """
    return {
        'lightgbm': LightGBMRanker,
        'xgboost': XGBoostRanker,
        'linear': LinearRanker,
    }

def create_model(name: str, **kwargs) -> BaseRanker:
    """Create a ranking model by name.
    
    Args:
        name: Name of the model ('lightgbm', 'xgboost', or 'linear')
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Instance of the specified model
        
    Raises:
        ValueError: If the model name is not recognized
    """
    models = available_models()
    
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Must be one of: {list(models.keys())}")
        
    return models[name](**kwargs)
