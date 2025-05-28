"""Evaluation metrics and utilities for ranking models."""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from scipy.stats import kendalltau, spearmanr

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Type aliases
ArrayLike = Union[np.ndarray, pd.Series, List[float]]
MetricFunction = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]

def _check_inputs(y_true: ArrayLike, y_pred: ArrayLike, qids: Optional[ArrayLike] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate and convert inputs to numpy arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
        
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
        
    if qids is not None:
        qids = np.asarray(qids)
        if qids.ndim != 1:
            raise ValueError("qids must be a 1D array")
            
        if len(qids) != len(y_true):
            raise ValueError("qids must have the same length as y_true and y_pred")
            
    return y_true, y_pred, qids

def _get_groups(qids: Optional[np.ndarray]) -> List[np.ndarray]:
    """Get indices for each query group."""
    if qids is None:
        return [np.arange(len(qids) if qids is not None else 0)]
        
    unique_qids = np.unique(qids)
    return [np.where(qids == qid)[0] for qid in unique_qids]

def _sort_by_score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort true and predicted scores by predicted score in descending order."""
    sort_idx = np.argsort(y_pred)[::-1]
    return y_true[sort_idx], y_pred[sort_idx]

def _get_discounts(n: int) -> np.ndarray:
    """Compute discount factors for DCG calculation."""
    return 1.0 / np.log2(np.arange(2, n + 2))

def dcg_score(y_true: ArrayLike, y_pred: ArrayLike, k: Optional[int] = None) -> float:
    """Compute Discounted Cumulative Gain (DCG) at position k.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted relevance scores
        k: Truncation level (use all if None)
        
    Returns:
        DCG score (higher is better)
    """
    y_true, y_pred, _ = _check_inputs(y_true, y_pred)
    
    if k is not None:
        k = min(k, len(y_true))
    else:
        k = len(y_true)
        
    if k == 0:
        return 0.0
        
    # Sort by predicted scores
    y_true_sorted, _ = _sort_by_score(y_true, y_pred)
    
    # Compute DCG
    discounts = _get_discounts(k)
    gains = 2**y_true_sorted[:k] - 1
    return np.sum(gains * discounts[:k])

def ndcg_score(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    qids: Optional[ArrayLike] = None,
    k: Optional[int] = None,
    top_k: Optional[int] = None,
    ignore_ties: bool = False
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG) at position k.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted relevance scores
        qids: Query IDs (for grouped computation)
        k: Truncation level (use all if None)
        top_k: If not None, only consider the top k predictions per query
        ignore_ties: If True, break ties randomly for consistent results
        
    Returns:
        NDCG score (higher is better, in [0, 1])
    """
    y_true, y_pred, qids = _check_inputs(y_true, y_pred, qids)
    
    if ignore_ties:
        # Add small random noise to break ties
        y_pred = y_pred.astype(float) + np.random.uniform(0, 1e-10, size=len(y_pred))
    
    if qids is None:
        qids = np.zeros_like(y_true)
        
    unique_qids = np.unique(qids)
    ndcgs = []
    
    for qid in unique_qids:
        mask = qids == qid
        y_true_q = y_true[mask]
        y_pred_q = y_pred[mask]
        
        if top_k is not None:
            top_k_idx = np.argsort(y_pred_q)[-top_k:]
            y_true_q = y_true_q[top_k_idx]
            y_pred_q = y_pred_q[top_k_idx]
        
        # Compute DCG
        dcg = dcg_score(y_true_q, y_pred_q, k)
        
        # Compute IDCG (ideal DCG)
        idcg = dcg_score(y_true_q, y_true_q, k)
        
        # Avoid division by zero
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
            
        ndcgs.append(ndcg)
    
    return float(np.mean(ndcgs))

def map_score(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    qids: Optional[ArrayLike] = None,
    k: Optional[int] = None,
    top_k: Optional[int] = None
) -> float:
    """Compute Mean Average Precision (MAP) at position k.
    
    Args:
        y_true: True relevance scores (binary or graded)
        y_pred: Predicted relevance scores
        qids: Query IDs (for grouped computation)
        k: Truncation level (use all if None)
        top_k: If not None, only consider the top k predictions per query
        
    Returns:
        MAP score (higher is better, in [0, 1])
    """
    y_true, y_pred, qids = _check_inputs(y_true, y_pred, qids)
    
    if qids is None:
        qids = np.zeros_like(y_true)
        
    unique_qids = np.unique(qids)
    aps = []
    
    for qid in unique_qids:
        mask = qids == qid
        y_true_q = y_true[mask]
        y_pred_q = y_pred[mask]
        
        if top_k is not None:
            top_k_idx = np.argsort(y_pred_q)[-top_k:]
            y_true_q = y_true_q[top_k_idx]
            y_pred_q = y_pred_q[top_k_idx]
        
        # Sort by predicted scores
        sort_idx = np.argsort(y_pred_q)[::-1]
        y_true_sorted = y_true_q[sort_idx]
        
        # Compute precision@k for each position
        rel_cumsum = np.cumsum(y_true_sorted > 0, dtype=float)
        precision_at_k = rel_cumsum / (np.arange(len(y_true_sorted)) + 1.0)
        
        # Compute average precision
        rel_mask = y_true_sorted > 0
        if np.any(rel_mask):
            ap = np.sum(precision_at_k * rel_mask) / np.sum(rel_mask)
            aps.append(ap)
        
    return float(np.mean(aps)) if aps else 0.0

def mrr_score(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    qids: Optional[ArrayLike] = None,
    k: Optional[int] = None,
    top_k: Optional[int] = None
) -> float:
    """Compute Mean Reciprocal Rank (MRR) at position k.
    
    Args:
        y_true: True relevance scores (binary or graded)
        y_pred: Predicted relevance scores
        qids: Query IDs (for grouped computation)
        k: Truncation level (use all if None)
        top_k: If not None, only consider the top k predictions per query
        
    Returns:
        MRR score (higher is better, in [0, 1])
    """
    y_true, y_pred, qids = _check_inputs(y_true, y_pred, qids)
    
    if qids is None:
        qids = np.zeros_like(y_true)
        
    unique_qids = np.unique(qids)
    rrs = []
    
    for qid in unique_qids:
        mask = qids == qid
        y_true_q = y_true[mask]
        y_pred_q = y_pred[mask]
        
        if top_k is not None:
            top_k_idx = np.argsort(y_pred_q)[-top_k:]
            y_true_q = y_true_q[top_k_idx]
            y_pred_q = y_pred_q[top_k_idx]
        
        # Sort by predicted scores
        sort_idx = np.argsort(y_pred_q)[::-1]
        y_true_sorted = y_true_q[sort_idx]
        
        # Find the first relevant document
        rel_positions = np.where(y_true_sorted > 0)[0]
        if len(rel_positions) > 0:
            first_rel_pos = rel_positions[0] + 1  # 1-based indexing
            rrs.append(1.0 / first_rel_pos)
    
    return float(np.mean(rrs)) if rrs else 0.0

def kendall_tau(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    qids: Optional[ArrayLike] = None,
    variant: str = 'b'
) -> float:
    """Compute Kendall's tau correlation between true and predicted rankings.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted relevance scores
        qids: Query IDs (for grouped computation)
        variant: Variant of Kendall's tau ('a', 'b', or 'c')
        
    Returns:
        Kendall's tau (in [-1, 1], higher is better)
    """
    y_true, y_pred, qids = _check_inputs(y_true, y_pred, qids)
    
    if qids is None:
        qids = np.zeros_like(y_true)
        
    unique_qids = np.unique(qids)
    taus = []
    
    for qid in unique_qids:
        mask = qids == qid
        y_true_q = y_true[mask]
        y_pred_q = y_pred[mask]
        
        if len(y_true_q) < 2:
            continue
            
        tau, _ = kendalltau(y_true_q, y_pred_q, variant=variant)
        if not np.isnan(tau):
            taus.append(tau)
    
    return float(np.mean(taus)) if taus else 0.0

def spearman_rho(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    qids: Optional[ArrayLike] = None
) -> float:
    """Compute Spearman's rank correlation coefficient.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted relevance scores
        qids: Query IDs (for grouped computation)
        
    Returns:
        Spearman's rho (in [-1, 1], higher is better)
    """
    y_true, y_pred, qids = _check_inputs(y_true, y_pred, qids)
    
    if qids is None:
        qids = np.zeros_like(y_true)
        
    unique_qids = np.unique(qids)
    rhos = []
    
    for qid in unique_qids:
        mask = qids == qid
        y_true_q = y_true[mask]
        y_pred_q = y_pred[mask]
        
        if len(y_true_q) < 2:
            continue
            
        rho, _ = spearmanr(y_true_q, y_pred_q)
        if not np.isnan(rho):
            rhos.append(rho)
    
    return float(np.mean(rhos)) if rhos else 0.0

def evaluate_ranking(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    qids: Optional[ArrayLike] = None,
    metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    return_dataframe: bool = True
) -> Union[Dict[str, float], pd.DataFrame]:
    """Evaluate ranking predictions with multiple metrics.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted relevance scores
        qids: Query IDs (for grouped computation)
        metrics: Dictionary of metric names to metric configurations
        return_dataframe: If True, return results as a pandas DataFrame
        
    Returns:
        Dictionary or DataFrame of metric scores
    """
    if metrics is None:
        metrics = {
            'ndcg@10': {'func': ndcg_score, 'k': 10},
            'map': {'func': map_score},
            'mrr': {'func': mrr_score},
            'kendall_tau': {'func': kendall_tau},
            'spearman_rho': {'func': spearman_rho}
        }
    
    results = {}
    
    for name, config in metrics.items():
        func = config['func']
        kwargs = {k: v for k, v in config.items() if k != 'func'}
        results[name] = func(y_true, y_pred, qids=qids, **kwargs)
    
    if return_dataframe:
        return pd.DataFrame([results])
    
    return results

def plot_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    qids: Optional[ArrayLike] = None,
    metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = 'Model Evaluation Metrics'
) -> go.Figure:
    """Create a bar plot of evaluation metrics.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted relevance scores
        qids: Query IDs (for grouped computation)
        metrics: Dictionary of metric names to metric configurations
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if metrics is None:
        metrics = {
            'NDCG@10': {'func': ndcg_score, 'k': 10},
            'MAP': {'func': map_score},
            'MRR': {'func': mrr_score},
            'Kendall\'s τ': {'func': kendall_tau},
            'Spearman\'s ρ': {'func': spearman_rho}
        }
    
    # Compute metrics
    results = evaluate_ranking(y_true, y_pred, qids, metrics, return_dataframe=False)
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(results.keys()),
        y=list(results.values()),
        text=[f'{v:.4f}' for v in results.values()],
        textposition='auto',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        template='plotly_white',
        height=500,
        width=800
    )
    
    return fig

# Scikit-learn compatible scorers
ndcg_scorer = make_scorer(ndcg_score, needs_threshold=False)
map_scorer = make_scorer(map_score, needs_threshold=False)
mrr_scorer = make_scorer(mrr_score, needs_threshold=False)
