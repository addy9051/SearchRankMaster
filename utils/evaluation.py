import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Import checks for ML libraries
from utils.model_training import check_lightgbm_available, check_xgboost_available

# We'll defer importing these libraries until needed
# TensorFlow has been removed from the application
TENSORFLOW_AVAILABLE = False  # We don't use TensorFlow anymore

@st.cache_data
def safe_calculate_ndcg(y_true, y_pred, k=10):
    """Safely calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        k: Cutoff for evaluation
        
    Returns:
        NDCG@k score, or 0.0 if calculation fails
    """
    try:
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if we have enough items
        n = min(len(y_true), len(y_pred), k)
        if n == 0:
            return 0.0
            
        # Calculate NDCG using the existing function but with safe inputs
        return calculate_ndcg(y_true, y_pred, k)
    except Exception as e:
        print(f"Error calculating NDCG: {str(e)}")
        return 0.0

@st.cache_data
def safe_calculate_map(y_true, y_pred, k=10):
    """Safely calculate Mean Average Precision (MAP) at k.
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        k: Cutoff for evaluation
        
    Returns:
        MAP@k score, or 0.0 if calculation fails
    """
    try:
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if we have enough items
        n = min(len(y_true), len(y_pred), k)
        if n == 0:
            return 0.0
            
        # Calculate MAP using the existing function but with safe inputs
        return calculate_map(y_true, y_pred, k)
    except Exception as e:
        print(f"Error calculating MAP: {str(e)}")
        return 0.0

@st.cache_data
def safe_calculate_mrr(y_true, y_pred):
    """Safely calculate Mean Reciprocal Rank (MRR).
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        
    Returns:
        MRR score, or 0.0 if calculation fails
    """
    try:
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if we have any items
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
            
        # Calculate MRR using the existing function but with safe inputs
        return calculate_mrr(y_true, y_pred)
    except Exception as e:
        print(f"Error calculating MRR: {str(e)}")
        return 0.0

@st.cache_data
def safe_calculate_precision_at_k(y_true, y_pred, k=10):
    """Safely calculate Precision at k.
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        k: Cutoff for evaluation
        
    Returns:
        Precision@k score, or 0.0 if calculation fails
    """
    try:
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if we have enough items
        n = min(len(y_true), len(y_pred), k)
        if n == 0:
            return 0.0
            
        # Calculate Precision@k using the existing function but with safe inputs
        return calculate_precision_at_k(y_true, y_pred, k)
    except Exception as e:
        print(f"Error calculating Precision@k: {str(e)}")
        return 0.0

@st.cache_data
def safe_calculate_recall_at_k(y_true, y_pred, k=10):
    """Safely calculate Recall at k.
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        k: Cutoff for evaluation
        
    Returns:
        Recall@k score, or 0.0 if calculation fails
    """
    try:
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if we have any relevant items
        total_relevant = np.sum(y_true > 0)
        if total_relevant == 0:
            return 0.0
            
        # Calculate Recall@k using the existing function but with safe inputs
        return calculate_recall_at_k(y_true, y_pred, k)
    except Exception as e:
        print(f"Error calculating Recall@k: {str(e)}")
        return 0.0


@st.cache_data
def calculate_ndcg(y_true, y_pred, k=10):
    """Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        k: Cutoff for evaluation
        
    Returns:
        NDCG@k score
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle empty inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
        
    # Ensure k is not larger than the number of documents
    k = min(k, len(y_true), len(y_pred))
    if k == 0:
        return 0.0
    
    try:
        # Get sorted indices based on predictions
        pred_indices = np.argsort(y_pred)[::-1]
        
        # Get true relevance scores in predicted order
        true_relevance = y_true[pred_indices]
        
        # Calculate DCG@k
        dcg = np.sum([true_relevance[i] / np.log2(i + 2) for i in range(min(k, len(true_relevance)))])
        
        # Calculate ideal DCG@k
        ideal_indices = np.argsort(y_true)[::-1]
        ideal_relevance = y_true[ideal_indices]
        idcg = np.sum([ideal_relevance[i] / np.log2(i + 2) for i in range(min(k, len(ideal_relevance)))])
        
        # Calculate NDCG@k
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    except Exception as e:
        print(f"Error in calculate_ndcg: {str(e)}")
        return 0.0

@st.cache_data
def calculate_map(y_true, y_pred, k=10):
    """Calculate Mean Average Precision (MAP) at k.
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        k: Cutoff for evaluation
        
    Returns:
        MAP@k score
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle empty inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
        
    # Ensure k is not larger than the number of documents
    k = min(k, len(y_true), len(y_pred))
    if k == 0:
        return 0.0
    
    try:
        # Get sorted indices based on predictions
        pred_indices = np.argsort(y_pred)[::-1][:k]
        
        # Convert relevance scores to binary relevance judgments (relevant if score > 0)
        binary_rel = (y_true > 0).astype(int)
        
        # Get binary relevance in predicted order
        true_relevance = binary_rel[pred_indices]
        
        # Calculate precision at each position
        precisions = []
        relevant_count = 0
        
        for i in range(min(len(true_relevance), k)):  # Ensure we don't go out of bounds
            if true_relevance[i] == 1:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        # Calculate average precision
        ap = np.mean(precisions) if precisions else 0.0
        
        return ap
    except Exception as e:
        print(f"Error in calculate_map: {str(e)}")
        return 0.0

@st.cache_data
def calculate_mrr(y_true, y_pred):
    """Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        
    Returns:
        MRR score
    """
    # Get sorted indices based on predictions
    pred_indices = np.argsort(y_pred)[::-1]
    
    # Convert relevance scores to binary relevance judgments (relevant if score > 0)
    binary_rel = (y_true > 0).astype(int)
    
    # Get binary relevance in predicted order
    true_relevance = binary_rel[pred_indices]
    
    # Find the first relevant document
    first_relevant = np.where(true_relevance == 1)[0]
    
    if len(first_relevant) > 0:
        # Calculate reciprocal rank (add 1 to index for rank)
        rr = 1.0 / (first_relevant[0] + 1)
    else:
        rr = 0.0
    
    return rr

@st.cache_data
def calculate_precision_at_k(y_true, y_pred, k=10):
    """Calculate Precision at k.
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        k: Cutoff for evaluation
        
    Returns:
        Precision@k score
    """
    # Get sorted indices based on predictions
    pred_indices = np.argsort(y_pred)[::-1][:k]
    
    # Convert relevance scores to binary relevance judgments (relevant if score > 0)
    binary_rel = (y_true > 0).astype(int)
    
    # Get binary relevance in predicted order
    true_relevance = binary_rel[pred_indices]
    
    # Calculate precision at k
    precision = np.mean(true_relevance)
    
    return precision

@st.cache_data
def calculate_recall_at_k(y_true, y_pred, k=10):
    """Calculate Recall at k.
    
    Args:
        y_true: Array of true relevance scores
        y_pred: Array of predicted relevance scores
        k: Cutoff for evaluation
        
    Returns:
        Recall@k score
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get sorted indices based on predictions
    pred_indices = np.argsort(y_pred)[::-1]
    
    # Get true relevance scores in predicted order
    true_relevance = y_true[pred_indices]
    
    # Calculate number of relevant items
    total_relevant = np.sum(y_true > 0)
    
    # If there are no relevant items, recall is 0
    if total_relevant == 0:
        return 0.0
    
    # Calculate recall@k
    relevant_at_k = np.sum(true_relevance[:k] > 0)
    recall = relevant_at_k / total_relevant
    
    return recall

@st.cache_data
def evaluate_model_by_query(_model, test_df, model_name, features_to_use=None):
    """Evaluate a model by query.
    
    Args:
        model: Trained model object
        test_df: Test data DataFrame
        model_name: Name of the model
        features_to_use: List of feature indices to use (if None, use all features)
        
    Returns:
        DataFrame with evaluation metrics per query
    """
    # Prepare feature columns
    print("\n=== Feature Selection Debug ===")
    print("features_to_use:", features_to_use)
    print("len(features_to_use):", len(features_to_use) if features_to_use is not None else "None")
    
    # Get all feature columns from the DataFrame
    available_feature_cols = [col for col in test_df.columns if str(col).startswith('feature_')]
    print(f"Found {len(available_feature_cols)} feature columns in the DataFrame")
    print("First 5 available feature columns:", available_feature_cols[:5])
    print("Last 5 available feature columns:", available_feature_cols[-5:] if len(available_feature_cols) > 5 else available_feature_cols)
    
    if features_to_use is not None and len(features_to_use) > 0:
        print("\nProcessing specified features...")
        # If features_to_use contains strings, use them directly after validation
        if all(isinstance(f, str) for f in features_to_use):
            print("Using string feature names directly")
            feature_cols = [f for f in features_to_use if f in available_feature_cols]
            print(f"Matched {len(feature_cols)} out of {len(features_to_use)} feature names")
        else:
            print("Processing numeric feature indices")
            # Handle numeric indices - assume 0-based indexing
            safe_indices = []
            for idx in features_to_use:
                try:
                    original_idx = idx
                    idx = int(idx)
                    print(f"  Processing index: {original_idx} (as int: {idx})")
                    
                    # Check if index is in 1-based range
                    if 1 <= idx <= len(available_feature_cols):
                        print(f"    - Treating as 1-based index, converting to 0-based: {idx-1}")
                        safe_indices.append(idx - 1)
                    # Check if index is in 0-based range
                    elif 0 <= idx < len(available_feature_cols):
                        print(f"    - Using as 0-based index: {idx}")
                        safe_indices.append(idx)
                    else:
                        print(f"    - Index {idx} out of range (0-{len(available_feature_cols)-1} for 0-based, 1-{len(available_feature_cols)} for 1-based)")
                        
                except (ValueError, TypeError) as e:
                    print(f"Skipping invalid feature index {original_idx}: {e}")
                    continue
            
            # Remove duplicates and sort
            safe_indices = sorted(list(set(safe_indices)))
            print(f"Final safe indices (0-based): {safe_indices}")
            # Convert to column names (1-based)
            feature_cols = [f'feature_{i+1}' for i in safe_indices]
    else:
        print("No features specified, using all available features")
        # Use all available features
        feature_cols = available_feature_cols
    
    # Ensure we only use columns that exist in the DataFrame
    final_feature_cols = [col for col in feature_cols if col in available_feature_cols]
    
    if len(final_feature_cols) != len(feature_cols):
        print(f"Warning: {len(feature_cols) - len(final_feature_cols)} specified features were not found in the DataFrame")
    
    if not final_feature_cols:
        raise ValueError("No valid feature columns found. Please check your feature selection.")
    
    print("\n=== Final Feature Selection ===")
    print(f"Using {len(final_feature_cols)} feature columns for evaluation")
    print("First 5 features:", final_feature_cols[:5])
    if len(final_feature_cols) > 5:
        print(f"... and {len(final_feature_cols)-5} more")
    
    # Verify the features exist in the DataFrame
    missing_cols = [col for col in final_feature_cols if col not in test_df.columns]
    if missing_cols:
        print(f"Warning: {len(missing_cols)} features are missing from the test DataFrame")
        print("Missing features:", missing_cols)
    
    feature_cols = final_feature_cols
    
    # Get unique query IDs
    query_ids = test_df['query_id'].unique()
    
    # Evaluation metrics to calculate
    metrics = {
        'ndcg@1': [],
        'ndcg@3': [],
        'ndcg@5': [],
        'ndcg@10': [],
        'map@10': [],
        'mrr': [],
        'precision@10': [],
        'recall@10': []
    }
    
    # Process each query
    for qid in query_ids:
        # Get data for this query
        query_df = test_df[test_df['query_id'] == qid]
        
        # Extract features and true relevance
        X = query_df[feature_cols].values
        y_true = query_df['relevance'].values
        
        # Get model predictions
        if model_name.startswith('LightGBM'):
            y_pred = _model.predict(X)
        elif model_name.startswith('XGBoost'):
            # Import XGBoost dynamically if needed
            if check_xgboost_available():
                import xgboost as xgb
                dtest = xgb.DMatrix(X)
                y_pred = _model.predict(dtest)
            else:
                st.error("XGBoost is not available for evaluation.")
                y_pred = np.zeros_like(y_true)
        elif model_name.startswith('TF-Ranking'):
            # TensorFlow is not supported anymore
            st.error("TensorFlow models are not supported in this version.")
            y_pred = np.zeros_like(y_true)
        else:
            y_pred = _model.predict(X)
        # Debug information for metric calculation
        print(f"DEBUG: qid={qid}, model_name='{model_name}'")
        print(f"  X.shape: {X.shape}, feature_cols count: {len(feature_cols)}")
        print(f"  y_true.shape: {y_true.shape}, y_true (first 5): {y_true[:5] if y_true is not None else 'None'}")
        print(f"  y_pred.shape: {y_pred.shape if y_pred is not None else 'None'}, y_pred (first 5): {y_pred[:5] if y_pred is not None else 'None'}")
        print(f"  len(query_df): {len(query_df)}")
        
        # Ensure y_pred is a numpy array and has the same length as y_true
        y_pred = np.asarray(y_pred)
        if len(y_pred) != len(y_true):
            print(f"WARNING: Length mismatch - y_true: {len(y_true)}, y_pred: {len(y_pred)}. Truncating to min length.")
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        # Log shapes for debugging
        print(f"  Calculating metrics for query {qid} with {len(y_true)} documents")
        print(f"  y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        
        # Calculate all metrics using safe functions
        metrics['ndcg@1'].append(safe_calculate_ndcg(y_true, y_pred, k=1))
        metrics['ndcg@3'].append(safe_calculate_ndcg(y_true, y_pred, k=3))
        metrics['ndcg@5'].append(safe_calculate_ndcg(y_true, y_pred, k=5))
        metrics['ndcg@10'].append(safe_calculate_ndcg(y_true, y_pred, k=10))
        metrics['map@10'].append(safe_calculate_map(y_true, y_pred, k=10))
        metrics['mrr'].append(safe_calculate_mrr(y_true, y_pred))
        metrics['precision@10'].append(safe_calculate_precision_at_k(y_true, y_pred, k=10))
        metrics['recall@10'].append(safe_calculate_recall_at_k(y_true, y_pred, k=10))
        
        # Log successful calculation
        print(f"  Successfully calculated all metrics for query {qid}")
    
    # Create DataFrame with results
    results_df = pd.DataFrame(metrics)
    
    # Add query ID and document count
    results_df['query_id'] = query_ids
    results_df['document_count'] = [len(test_df[test_df['query_id'] == qid]) for qid in query_ids]
    
    # Calculate overall metrics (mean across queries)
    overall_metrics = results_df.drop(['query_id', 'document_count'], axis=1).mean().to_dict()
    
    return results_df, overall_metrics

def get_feature_importance(model, model_name, feature_count=136, features_to_use=None):
    """Extract feature importance from the trained model.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        feature_count: Total number of features in the dataset
        features_to_use: List of feature indices that were used in training
        
    Returns:
        DataFrame with feature importance scores
    """
    try:
        # Initialize importance array with zeros
        importance = np.zeros(feature_count)
        
        # Extract feature importance based on model type
        if model_name.startswith('Random Forest'):
            if hasattr(model, 'feature_importances_'):
                if features_to_use is not None and len(features_to_use) == len(model.feature_importances_):
                    # Map feature importances back to their original indices
                    for i, idx in enumerate(features_to_use):
                        if 0 <= idx < feature_count:
                            importance[idx] = model.feature_importances_[i]
                else:
                    # Use feature importances as is if no feature mapping is provided
                    n_features = min(len(model.feature_importances_), feature_count)
                    importance[:n_features] = model.feature_importances_[:n_features]
        
        elif model_name.startswith('LightGBM'):
            if check_lightgbm_available() and hasattr(model, 'feature_importance'):
                try:
                    model_importance = model.feature_importance(importance_type='gain')
                    if features_to_use is not None and len(features_to_use) == len(model_importance):
                        # Map feature importances back to their original indices
                        for i, idx in enumerate(features_to_use):
                            if 0 <= idx < feature_count:
                                importance[idx] = model_importance[i]
                    else:
                        # Use feature importances as is if no feature mapping is provided
                        n_features = min(len(model_importance), feature_count)
                        importance[:n_features] = model_importance[:n_features]
                except Exception as e:
                    st.error(f"Error getting LightGBM feature importance: {str(e)}")
        
        elif model_name.startswith('XGBoost'):
            if check_xgboost_available() and hasattr(model, 'get_score'):
                try:
                    importance_dict = model.get_score(importance_type='gain')
                    if features_to_use is not None:
                        # Map feature importances back to their original indices
                        for i, idx in enumerate(features_to_use):
                            if 0 <= idx < feature_count and f'f{i}' in importance_dict:
                                importance[idx] = importance_dict[f'f{i}']
                    else:
                        # Convert dictionary to array
                        for f_name, imp in importance_dict.items():
                            try:
                                idx = int(f_name[1:])  # Extract index from feature name (e.g., 'f0' -> 0)
                                if 0 <= idx < feature_count:
                                    importance[idx] = imp
                            except (ValueError, IndexError):
                                continue
                except Exception as e:
                    st.error(f"Error getting XGBoost feature importance: {str(e)}")
        
        elif model_name.startswith(('Linear Regression', 'Ridge', 'Lasso')):
            if hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:  # For multi-class classification
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)
                    
                if features_to_use is not None and len(features_to_use) == len(coef):
                    # Map coefficients back to their original indices
                    for i, idx in enumerate(features_to_use):
                        if 0 <= idx < feature_count:
                            importance[idx] = coef[i]
                else:
                    # Use coefficients as is if no feature mapping is provided
                    n_features = min(len(coef), feature_count)
                    importance[:n_features] = coef[:n_features]
        
        # Create DataFrame with feature importance
        importance_df = pd.DataFrame({
            'feature_index': list(range(feature_count)),
            'importance': importance
        })
        
        # Normalize importance scores (if any non-zero values)
        if np.sum(importance) > 0:
            importance_df['importance'] = importance_df['importance'] / np.sum(importance_df['importance'])
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        st.error(f"Error calculating feature importance: {str(e)}")
        # Return empty DataFrame with correct structure on error
        return pd.DataFrame({
            'feature_index': list(range(feature_count)),
            'importance': np.zeros(feature_count)
        }).sort_values('importance', ascending=False)

def plot_metric_distribution(results_df, metric_name, use_plotly=True):
    """Plot the distribution of a metric across queries.
    
    Args:
        results_df: DataFrame with evaluation metrics per query
        metric_name: Name of the metric to plot
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    if use_plotly:
        fig = px.histogram(
            results_df,
            x=metric_name,
            marginal='box',
            title=f'Distribution of {metric_name.upper()} across Queries',
            labels={metric_name: metric_name.upper()},
            nbins=20
        )
        
        fig.update_layout(
            xaxis_title=metric_name.upper(),
            yaxis_title='Frequency',
            height=500
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(results_df[metric_name], bins=20, alpha=0.7)
        
        # Add mean line
        mean_value = results_df[metric_name].mean()
        ax.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.4f}')
        
        ax.set_title(f'Distribution of {metric_name.upper()} across Queries')
        ax.set_xlabel(metric_name.upper())
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        return plt

def plot_metrics_comparison(metrics_list, model_names, use_plotly=True):
    """Plot a comparison of metrics across models.
    
    Args:
        metrics_list: List of dictionaries with metrics for each model
        model_names: List of model names
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    # Create a DataFrame for comparison
    comparison_data = []
    
    for i, metrics in enumerate(metrics_list):
        model_name = model_names[i]
        
        for metric, value in metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': value
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if use_plotly:
        fig = px.bar(
            comparison_df,
            x='Metric',
            y='Value',
            color='Model',
            barmode='group',
            title='Metrics Comparison Across Models',
            labels={'Value': 'Score'},
            height=600
        )
        
        fig.update_layout(
            xaxis={'categoryorder': 'category ascending'},
            yaxis_range=[0, 1]
        )
        return fig
    else:
        # Pivot the DataFrame for easier plotting
        pivot_df = comparison_df.pivot(index='Metric', columns='Model', values='Value')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart
        pivot_df.plot(kind='bar', ax=ax)
        
        ax.set_title('Metrics Comparison Across Models')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend(title='Model')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plt

def plot_training_history(history, model_name, use_plotly=True):
    """Plot training history metrics.
    
    Args:
        history: Dictionary with training history
        model_name: Name of the model
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    if use_plotly:
        fig = go.Figure()
        
        if 'train_loss' in history and history['train_loss']:
            fig.add_trace(go.Scatter(
                y=history['train_loss'],
                mode='lines',
                name='Training Loss'
            ))
        
        if 'val_loss' in history and history['val_loss']:
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='Validation Loss'
            ))
        
        # Check if we have ranking metrics
        if 'train_metrics' in history and 'val_metrics' in history:
            for dataset in ['train', 'val']:
                metrics_dict = history[f'{dataset}_metrics']
                
                for metric_name, values in metrics_dict.items():
                    # Skip metrics that are not iterative (e.g., single values)
                    if isinstance(values, list) and len(values) > 1:
                        fig.add_trace(go.Scatter(
                            y=values,
                            mode='lines',
                            name=f'{dataset.capitalize()} {metric_name}'
                        ))
        
        fig.update_layout(
            title=f'Training History for {model_name}',
            xaxis_title='Iteration',
            yaxis_title='Value',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'train_loss' in history and history['train_loss']:
            ax.plot(history['train_loss'], label='Training Loss')
        
        if 'val_loss' in history and history['val_loss']:
            ax.plot(history['val_loss'], label='Validation Loss')
        
        # Check if we have ranking metrics
        if 'train_metrics' in history and 'val_metrics' in history:
            for dataset in ['train', 'val']:
                metrics_dict = history[f'{dataset}_metrics']
                
                for metric_name, values in metrics_dict.items():
                    # Skip metrics that are not iterative (e.g., single values)
                    if isinstance(values, list) and len(values) > 1:
                        ax.plot(values, label=f'{dataset.capitalize()} {metric_name}')
        
        ax.set_title(f'Training History for {model_name}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.legend()
        
        plt.tight_layout()
        return plt

def plot_feature_importance_comparison(importance_dfs, model_names, top_n=10, use_plotly=True):
    """Plot a comparison of feature importance across models.
    
    Args:
        importance_dfs: List of DataFrames with feature importance for each model
        model_names: List of model names
        top_n: Number of top features to include
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    # Create a set of all top features across models
    top_features = set()
    
    for df in importance_dfs:
        top_features.update(df.head(top_n)['feature_index'].values)
    
    # Convert to list and sort
    top_features = sorted(list(top_features))
    
    # Create comparison data
    comparison_data = []
    
    for i, df in enumerate(importance_dfs):
        model_name = model_names[i]
        
        # Create a dictionary mapping feature index to importance
        importance_dict = df.set_index('feature_index')['importance'].to_dict()
        
        for feat_idx in top_features:
            comparison_data.append({
                'Model': model_name,
                'Feature': f'Feature {feat_idx+1}',
                'Importance': importance_dict.get(feat_idx, 0.0)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if use_plotly:
        fig = px.bar(
            comparison_df,
            x='Feature',
            y='Importance',
            color='Model',
            barmode='group',
            title=f'Feature Importance Comparison Across Models (Top {len(top_features)} Features)',
            labels={'Importance': 'Normalized Importance'},
            height=600
        )
        
        fig.update_layout(
            xaxis={'categoryorder': 'category ascending'}
        )
        return fig
    else:
        # Pivot the DataFrame for easier plotting
        pivot_df = comparison_df.pivot(index='Feature', columns='Model', values='Importance')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart
        pivot_df.plot(kind='bar', ax=ax)
        
        ax.set_title(f'Feature Importance Comparison Across Models (Top {len(top_features)} Features)')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Normalized Importance')
        ax.legend(title='Model')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plt

@st.cache_data
def get_predictions_by_query(_model, test_df, model_name, query_id, features_to_use=None):
    """Get model predictions for a specific query.
    
    Args:
        model: Trained model object
        test_df: Test data DataFrame
        model_name: Name of the model
        query_id: ID of the query to get predictions for
        features_to_use: List of feature indices to use (if None, use all features)
        
    Returns:
        DataFrame with document information and model predictions
    """
    try:
        # Debug: Print available columns in the test DataFrame
        print("Available columns in test_df:", test_df.columns.tolist())
        
        # Filter data for the specific query
        query_df = test_df[test_df['query_id'] == query_id].copy()
        
        if query_df.empty:
            st.warning(f"No data found for query_id: {query_id}")
            return None
        
        # Debug: Print query_df columns and first few rows
        print(f"Columns in query_df for query {query_id}:", query_df.columns.tolist())
        print("First few rows of query_df:", query_df.head())
        
        # Prepare feature columns
        print("features_to_use:", features_to_use)
        print("len(features_to_use):", len(features_to_use) if features_to_use is not None else "None")
        
        # Determine available feature columns in the DataFrame
        available_feature_cols = [col for col in test_df.columns if col.startswith('feature_')]
        print(f"Found {len(available_feature_cols)} feature columns in the DataFrame")
        
        if features_to_use is not None:
            # Filter out any indices that would be out of bounds
            safe_indices = [i for i in features_to_use if i >= 1 and i <= 136]
            
            # Create feature column names
            feature_cols = [f'feature_{i}' for i in safe_indices]
        else:
            # Use all features
            feature_cols = [f'feature_{i}' for i in range(1, 137)]

        # Ensure we have the correct columns in the DataFrame
        feature_cols = [col for col in feature_cols if col in query_df.columns]
        print("Final feature_cols length:", len(feature_cols))
        
        if len(feature_cols) == 0:
            st.error("No valid feature columns found in the DataFrame")
            return None
            
        # Extract features
        X = query_df[feature_cols].values
        
        # Get model predictions
        if model_name.startswith('LightGBM'):
            query_df['prediction'] = _model.predict(X)
        elif model_name.startswith('XGBoost'):
            # Import XGBoost dynamically if needed
            if check_xgboost_available():
                import xgboost as xgb
                dtest = xgb.DMatrix(X)
                query_df['prediction'] = _model.predict(dtest)
            else:
                st.error("XGBoost is not available for predictions.")
                query_df['prediction'] = np.zeros(len(query_df))
        elif model_name.startswith('TF-Ranking'):
            # TensorFlow is not supported anymore
            st.error("TensorFlow models are not supported in this version.")
            query_df['prediction'] = np.zeros(len(query_df))
        else:
            query_df['prediction'] = _model.predict(X)
        
        # Sort by prediction score (descending)
        query_df = query_df.sort_values('prediction', ascending=False)
        
        # Prepare result with available columns
        result_columns = []
        
        # Add document identifier if available
        doc_id_cols = [col for col in ['document_id', 'doc_id', 'id'] if col in query_df.columns]
        if doc_id_cols:
            result_columns.append(doc_id_cols[0])
        
        # Add relevance if available
        if 'relevance' in query_df.columns:
            result_columns.append('relevance')
        
        # Add prediction
        result_columns.append('prediction')
        
        # If no identifier column is found, use the index
        if not doc_id_cols:
            query_df = query_df.reset_index()
            if 'index' in query_df.columns:
                result_columns.insert(0, 'index')
        
        # Select only the available columns
        result_df = query_df[result_columns].copy()
        
        # Rename columns for consistency
        if 'index' in result_df.columns:
            result_df = result_df.rename(columns={'index': 'document_id'})
        
        print("Result DataFrame columns:", result_df.columns.tolist())
        return result_df
        
    except Exception as e:
        st.error(f"Error getting predictions for query {query_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
