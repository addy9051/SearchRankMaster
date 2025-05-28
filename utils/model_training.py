import numpy as np
import pandas as pd
import streamlit as st
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# Set up flags to check library availability, but don't import them yet
LIGHTGBM_AVAILABLE = False
XGBOOST_AVAILABLE = False

# Define checks for library availability without importing at the module level
def check_lightgbm_available():
    """Check if LightGBM is available without importing it at the module level."""
    global LIGHTGBM_AVAILABLE
    if not LIGHTGBM_AVAILABLE:
        try:
            # Only attempt to import when actually checking
            import lightgbm
            LIGHTGBM_AVAILABLE = True
        except (ImportError, OSError):
            LIGHTGBM_AVAILABLE = False
            st.warning("LightGBM is not available. Some models will be disabled.")
    return LIGHTGBM_AVAILABLE

def check_xgboost_available():
    """Check if XGBoost is available without importing it at the module level."""
    global XGBOOST_AVAILABLE
    if not XGBOOST_AVAILABLE:
        try:
            # Only attempt to import when actually checking
            import xgboost
            XGBOOST_AVAILABLE = True
        except (ImportError, OSError):
            XGBOOST_AVAILABLE = False
            st.warning("XGBoost is not available. Some models will be disabled.")
    return XGBOOST_AVAILABLE

from utils.data_loader import prepare_dataset_for_training

# Dictionary of available models - will be filtered based on available libraries
def get_available_models():
    """Return the models that are available based on installed libraries."""
    models = {
        'Pointwise': [
            'Linear Regression (Ridge)',
            'Random Forest Regressor'
        ],
        'Pairwise': [],
        'Listwise': []
    }
    
    # Add models based on available libraries
    if check_lightgbm_available():
        models['Pointwise'].append('LightGBM Regressor')
        models['Pairwise'].append('LightGBM Ranker')
        models['Listwise'].append('LightGBM Ranker (listwise)')
    
    if check_xgboost_available():
        models['Pointwise'].append('XGBoost Regressor')
        models['Pairwise'].append('XGBoost Ranker')
    
    # Clean up empty categories
    return {k: v for k, v in models.items() if v}

def get_model_approach(model_name):
    """Get the approach (pointwise, pairwise, listwise) for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Approach name (Pointwise, Pairwise, or Listwise)
    """
    available_models = get_available_models()
    for approach, models in available_models.items():
        if model_name in models:
            return approach
    return None

def get_model_params(model_name):
    """Get default parameters for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of default parameters
    """
    # Default parameters for each model
    if model_name == 'Linear Regression (Ridge)':
        return {
            'alpha': 1.0,
            'max_iter': 1000
        }
    elif model_name == 'Random Forest Regressor':
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    elif model_name == 'LightGBM Regressor':
        return {
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'regression',
            'random_state': 42
        }
    elif model_name == 'XGBoost Regressor':
        return {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    elif model_name == 'LightGBM Ranker':
        return {
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'random_state': 42
        }
    elif model_name == 'XGBoost Ranker':
        return {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'rank:pairwise',
            'random_state': 42
        }
    elif model_name == 'LightGBM Ranker (listwise)':
        return {
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'random_state': 42
        }
    # TensorFlow models have been removed
    else:
        return {}

def train_model(train_df, val_df, model_name, model_params, features_to_use=None, progress_bar=None):
    """Train a ranking model.
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        model_name: Name of the model to train
        model_params: Dictionary of model parameters
        features_to_use: List of feature indices to use (if None, use all features)
        progress_bar: Streamlit progress bar object
        
    Returns:
        Trained model and training history/metrics
    """
    # Prepare data
    X_train, y_train, qids_train = prepare_dataset_for_training(train_df, features_to_use)
    X_val, y_val, qids_val = prepare_dataset_for_training(val_df, features_to_use)
    
    # Training history for metrics - initialize with empty lists/dicts for all possible metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': {},
        'val_metrics': {},
        'epochs': 0
    }
    
    # Update progress bar
    if progress_bar:
        progress_bar.progress(0.1, text="Prepared data for training")
    
    if model_name == 'Linear Regression (Ridge)':
        # Add a small positive value to alpha to ensure numerical stability
        alpha = max(1e-3, float(model_params.get('alpha', 1.0)))
        model = Ridge(
            alpha=alpha,
            max_iter=model_params.get('max_iter', 1000),
            solver='svd'  # More numerically stable solver
        )
        model.fit(X_train, y_train)
        
        # Simple evaluation metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Store MSE as loss
        train_loss = float(np.mean((train_pred - y_train) ** 2))
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        
        # For non-iterative models, we'll simulate epochs for visualization
        history['train_loss'] = [train_loss] * 10  # Repeat for 10 epochs for visualization
        history['val_loss'] = [val_loss] * 10
        history['epochs'] = 10
        
        if progress_bar:
            progress_bar.progress(1.0, text="Training completed")
    
    elif model_name == 'Random Forest Regressor':
        model = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            min_samples_split=model_params.get('min_samples_split', 2),
            min_samples_leaf=model_params.get('min_samples_leaf', 1),
            random_state=model_params.get('random_state', 42)
        )
        
        if progress_bar:
            progress_bar.progress(0.2, text="Training Random Forest model")
        
        model.fit(X_train, y_train)
        
        # Simple evaluation metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Store MSE as loss
        train_loss = float(np.mean((train_pred - y_train) ** 2))
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        
        # For non-iterative models, we'll simulate epochs for visualization
        history['train_loss'] = [train_loss] * 10  # Repeat for 10 epochs for visualization
        history['val_loss'] = [val_loss] * 10
        history['epochs'] = 10
        
        if progress_bar:
            progress_bar.progress(1.0, text="Training completed")
    
    elif model_name == 'LightGBM Regressor':
        # Check if LightGBM is available
        if not check_lightgbm_available():
            st.error("LightGBM is not available. Please install it or choose another model.")
            if progress_bar:
                progress_bar.progress(1.0, text="Error: LightGBM not available")
            return None, None
            
        # Import LightGBM dynamically
        import lightgbm as lgb
        
        # Create the LightGBM datasets
        lgb_train = lgb.Dataset(X_train, y_train, group=None)
        lgb_val = lgb.Dataset(X_val, y_val, group=None, reference=lgb_train)
        
        # Get parameters
        params = {
            'objective': 'regression',
            'num_leaves': model_params.get('num_leaves', 31),
            'learning_rate': model_params.get('learning_rate', 0.1),
            'max_depth': model_params.get('max_depth', -1),
            'verbose': -1
        }
        
        if progress_bar:
            progress_bar.progress(0.2, text="Training LightGBM model")
        
        # Train the model
        evals_result = {}
        try:
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=model_params.get('n_estimators', 100),
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'val'],
                callbacks = [
                    lgb.record_evaluation(evals_result),
                    lgb.early_stopping(10),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # Get training history
            history['train_loss'] = evals_result['train']['l2']
            history['val_loss'] = evals_result['val']['l2']
        except Exception as e:
            st.error(f"Error training LightGBM model: {str(e)}")
            if progress_bar:
                progress_bar.progress(1.0, text="Error training model")
            return None, None
        
        if progress_bar:
            progress_bar.progress(1.0, text="Training completed")
    
    elif model_name == 'XGBoost Regressor':
        # Check if XGBoost is available
        if not check_xgboost_available():
            st.error("XGBoost is not available. Please install it or choose another model.")
            if progress_bar:
                progress_bar.progress(1.0, text="Error: XGBoost not available")
            return None, None
            
        # Import XGBoost dynamically
        import xgboost as xgb
        
        try:
            # Create the XGBoost DMatrix datasets
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Get parameters
            params = {
                'objective': 'reg:squarederror',
                'max_depth': model_params.get('max_depth', 6),
                'learning_rate': model_params.get('learning_rate', 0.1),
                'eval_metric': 'rmse'
            }
            
            if progress_bar:
                progress_bar.progress(0.2, text="Training XGBoost model")
            
            # Train the model
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=model_params.get('n_estimators', 100),
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=10,
                evals_result=evals_result,
                verbose_eval=False
            )
            
            # Get training history
            history['train_metrics'] = evals_result['train']
            history['val_metrics'] = evals_result['val']
            
            # For XGBoost, use RMSE as loss if available
            if 'rmse' in evals_result['train']:
                history['train_loss'] = evals_result['train']['rmse']
                history['val_loss'] = evals_result['val']['rmse']
            elif len(evals_result['train']) > 0:
                first_metric = next(iter(evals_result['train'].keys()))
                history['train_loss'] = evals_result['train'][first_metric]
                history['val_loss'] = evals_result['val'][first_metric]
                
            history['epochs'] = len(history['train_loss'])
        except Exception as e:
            st.error(f"Error training XGBoost model: {str(e)}")
            if progress_bar:
                progress_bar.progress(1.0, text="Error training model")
            return None, None
        
        if progress_bar:
            progress_bar.progress(1.0, text="Training completed")
    
    elif model_name == 'LightGBM Ranker' or model_name == 'LightGBM Ranker (listwise)':
        # Check if LightGBM is available
        if not check_lightgbm_available():
            st.error("LightGBM is not available. Please install it or choose another model.")
            if progress_bar:
                progress_bar.progress(1.0, text="Error: LightGBM not available")
            return None, None
            
        # Import LightGBM dynamically
        import lightgbm as lgb
        
        try:
            # Convert qids to group sizes for LightGBM
            train_groups = np.array([len(qids_train[qids_train == qid]) for qid in np.unique(qids_train)])
            val_groups = np.array([len(qids_val[qids_val == qid]) for qid in np.unique(qids_val)])
            
            # Create the LightGBM datasets
            lgb_train = lgb.Dataset(X_train, y_train, group=train_groups)
            lgb_val = lgb.Dataset(X_val, y_val, group=val_groups, reference=lgb_train)
            
            # Get parameters
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': model_params.get('ndcg_eval_at', [1, 3, 5, 10]),
                'num_leaves': model_params.get('num_leaves', 31),
                'learning_rate': model_params.get('learning_rate', 0.1),
                'max_depth': model_params.get('max_depth', -1),
                'verbose': -1
            }
            
            if progress_bar:
                progress_bar.progress(0.2, text="Training LightGBM Ranker model")
            
            # Train the model
            evals_result = {}
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=model_params.get('n_estimators', 100),
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'val'],
                callbacks = [
                    lgb.record_evaluation(evals_result),
                    lgb.early_stopping(10),
                    lgb.log_evaluation(period=0)
                ],
            )
            
            # Get training history
            history['train_metrics'] = evals_result['train']
            history['val_metrics'] = evals_result['val']
        except Exception as e:
            st.error(f"Error training LightGBM Ranker model: {str(e)}")
            if progress_bar:
                progress_bar.progress(1.0, text="Error training model")
            return None, None
        
        if progress_bar:
            progress_bar.progress(1.0, text="Training completed")
    
    elif model_name == 'XGBoost Ranker':
        # Check if XGBoost is available
        if not check_xgboost_available():
            st.error("XGBoost is not available. Please install it or choose another model.")
            if progress_bar:
                progress_bar.progress(1.0, text="Error: XGBoost not available")
            return None, None
            
        # Import XGBoost dynamically
        import xgboost as xgb
        
        try:
            # Create the XGBoost DMatrix datasets
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Set query group info
            train_groups = []
            for qid in np.unique(qids_train):
                train_groups.append(np.sum(qids_train == qid))
            
            val_groups = []
            for qid in np.unique(qids_val):
                val_groups.append(np.sum(qids_val == qid))
            
            dtrain.set_group(train_groups)
            dval.set_group(val_groups)
            
            # Get parameters
            params = {
                'objective': 'rank:pairwise',
                'max_depth': model_params.get('max_depth', 6),
                'learning_rate': model_params.get('learning_rate', 0.1),
                'eval_metric': ['ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10']
            }
            
            if progress_bar:
                progress_bar.progress(0.2, text="Training XGBoost Ranker model")
            
            # Train the model
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=model_params.get('n_estimators', 100),
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=10,
                evals_result=evals_result,
                verbose_eval=False
            )
            
            # Get training history
            history['train_metrics'] = evals_result['train']
            history['val_metrics'] = evals_result['val']
        except Exception as e:
            st.error(f"Error training XGBoost Ranker model: {str(e)}")
            if progress_bar:
                progress_bar.progress(1.0, text="Error training model")
            return None, None
        
        if progress_bar:
            progress_bar.progress(1.0, text="Training completed")
    
    # TensorFlow models have been removed from this application due to environment constraints
    # The application now focuses on traditional ML models like LightGBM and XGBoost
    else:
        st.error(f"Model {model_name} is not supported in this version of the application.")
        model = None
        history = None
        if progress_bar:
            progress_bar.progress(1.0, text="Model not supported")
    
    return model, history
