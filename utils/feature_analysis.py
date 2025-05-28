import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import seaborn as sns
from utils.data_loader import get_feature_group, FEATURE_GROUPS

@st.cache_data
def calculate_feature_importance(df, method='mutual_info'):
    """Calculate feature importance scores.
    
    Args:
        df: DataFrame with features and relevance
        method: Method to use ('mutual_info' or 'correlation')
        
    Returns:
        DataFrame with feature indices and importance scores
    """
    # Get all feature columns (f0, f1, etc.)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    feature_cols.sort(key=lambda x: int(x[8:]))  # Sort by feature number
    
    if not feature_cols:
        raise ValueError("No feature columns found in the dataset. Expected columns like 'f0', 'f1', etc.")
        
    X = df[feature_cols].values
    y = df['relevance'].values
    
    importance_scores = []
    
    if method == 'mutual_info':
        # Calculate mutual information between each feature and relevance
        mi_scores = mutual_info_regression(X, y, random_state=42)
        for i, score in enumerate(mi_scores):
            # Convert to 1-based index for get_feature_group
            importance_scores.append({
                'feature_index': i,
                'feature_name': feature_cols[i],
                'importance': score, 
                'group': get_feature_group(int(feature_cols[i][8:]))
            })
    
    elif method == 'correlation':
        # Calculate absolute Spearman correlation between each feature and relevance
        for i in range(X.shape[1]):
            corr, _ = spearmanr(X[:, i], y)
            # Use absolute correlation as importance
            importance_scores.append({
                'feature_index': i,
                'feature_name': feature_cols[i],
                'importance': abs(corr), 
                'group': get_feature_group(int(feature_cols[i][8:]))
            })
            
    return pd.DataFrame(importance_scores).sort_values('importance', ascending=False)

@st.cache_data
def calculate_feature_correlations(df, top_n=20):
    """Calculate correlations between top features.
    
    Args:
        df: DataFrame with features
        top_n: Number of top features to include
        
    Returns:
        Correlation matrix for top_n features
    """
    # Calculate feature importance first
    importance_df = calculate_feature_importance(df)
    
    # Get the top n important features
    top_features = importance_df.head(top_n)['feature_index'].values
    
    # Select these features from the DataFrame
    feature_cols = [f'feature_{i}' for i in top_features]
    X_top = df[feature_cols]
    
    # Calculate correlation matrix
    corr_matrix = X_top.corr(method='spearman')
    
    # Rename columns and index for better readability
    feature_names = [f'Feature {i+1}' for i in top_features]
    corr_matrix.columns = feature_names
    corr_matrix.index = feature_names
    
    return corr_matrix

@st.cache_data
def analyze_feature_distributions(df, features_to_analyze):
    """Analyze distributions of selected features.
    
    Args:
        df: DataFrame with features
        features_to_analyze: List of feature indices to analyze
        
    Returns:
        Dictionary with distribution statistics
    """
    results = {}
    
    for feature_idx in features_to_analyze:
        feature_col = f'feature_{feature_idx}'
        feature_data = df[feature_col].values
        
        # Calculate basic statistics
        results[feature_idx] = {
            'mean': np.mean(feature_data),
            'median': np.median(feature_data),
            'std': np.std(feature_data),
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'percentiles': {
                '25%': np.percentile(feature_data, 25),
                '75%': np.percentile(feature_data, 75),
                '95%': np.percentile(feature_data, 95)
            }
        }
    
    return results

@st.cache_data
def analyze_feature_by_relevance(df, feature_idx):
    """Analyze how a feature varies across different relevance levels.
    
    Args:
        df: DataFrame with features and relevance
        feature_idx: Index of the feature to analyze
        
    Returns:
        DataFrame with feature statistics per relevance level
    """
    feature_col = f'feature_{feature_idx}'
    
    # Group by relevance and calculate statistics
    grouped = df.groupby('relevance')[feature_col].agg(['mean', 'median', 'std', 'count'])
    
    # Reset index to make 'relevance' a regular column
    grouped = grouped.reset_index()
    
    return grouped

def plot_feature_importance(importance_df, top_n=20, use_plotly=True):
    """Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance scores
        top_n: Number of top features to show
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    try:
        # Take top N features
        plot_df = importance_df.head(top_n).copy()
        
        # Create feature names if not already present
        if 'feature_name' not in plot_df.columns:
            plot_df['feature_name'] = plot_df['feature_index'].apply(lambda x: f'Feature {x+1}')
        
        if use_plotly:
            # Check if 'group' column exists for coloring
            if 'group' in plot_df.columns:
                fig = px.bar(
                    plot_df, 
                    x='importance', 
                    y='feature_name',
                    color='group',
                    orientation='h',
                    labels={
                        'importance': 'Importance Score', 
                        'feature_name': 'Feature', 
                        'group': 'Feature Group'
                    },
                    title=f'Top {top_n} Feature Importance'
                )
            else:
                fig = px.bar(
                    plot_df, 
                    x='importance', 
                    y='feature_name',
                    orientation='h',
                    labels={
                        'importance': 'Importance Score', 
                        'feature_name': 'Feature'
                    },
                    title=f'Top {top_n} Feature Importance',
                    color_discrete_sequence=['#1f77b4']  # Default blue color
                )
            
            fig.update_layout(
                height=600, 
                yaxis={'categoryorder': 'total ascending'},
                showlegend=('group' in plot_df.columns)  # Only show legend if group exists
            )
            return fig
        else:
            # Create a Matplotlib bar chart
            plt.figure(figsize=(10, 12))
            if 'group' in plot_df.columns:
                sns.barplot(x='importance', y='feature_name', hue='group', data=plot_df)
            else:
                sns.barplot(x='importance', y='feature_name', data=plot_df, color='#1f77b4')
            
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.tight_layout()
            return plt
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_feature_correlation_heatmap(corr_matrix, use_plotly=True):
    """Plot correlation heatmap for top features.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    if use_plotly:
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Feature Correlation Heatmap'
        )
        fig.update_layout(height=700, width=700)
        return fig
    else:
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   mask=mask, vmin=-1, vmax=1, center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        return plt

def plot_feature_distribution(df, feature_idx, use_plotly=True):
    """Plot distribution of a feature.
    
    Args:
        df: DataFrame with features
        feature_idx: Index of the feature to plot
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    feature_col = f'feature_{feature_idx}'
    feature_data = df[feature_col].values
    feature_name = f'Feature {feature_idx}'
    group_name = get_feature_group(feature_idx)
    
    if use_plotly:
        fig = px.histogram(
            x=feature_data,
            nbins=50,
            title=f'Distribution of {feature_name} ({group_name})',
            labels={'x': feature_name, 'y': 'Frequency'},
        )
        
        # Add box plot as a second subplot
        fig2 = px.box(x=feature_data, title=f'Box Plot of {feature_name}')
        fig.add_trace(fig2.data[0])
        
        fig.update_layout(
            xaxis_title=feature_name,
            yaxis_title="Frequency",
            height=500
        )
        return fig
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Histogram
        sns.histplot(feature_data, bins=50, kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {feature_name} ({group_name})')
        ax1.set_xlabel(feature_name)
        ax1.set_ylabel('Frequency')
        
        # Box plot
        sns.boxplot(x=feature_data, ax=ax2)
        ax2.set_title(f'Box Plot of {feature_name}')
        ax2.set_xlabel(feature_name)
        
        plt.tight_layout()
        return plt

def plot_feature_by_relevance(grouped_df, feature_idx, use_plotly=True):
    """Plot feature statistics by relevance level.
    
    Args:
        grouped_df: DataFrame with feature statistics per relevance level
        feature_idx: Index of the feature being plotted
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    feature_name = f'Feature {feature_idx}'
    group_name = get_feature_group(feature_idx)     
    
    if use_plotly:
        fig = go.Figure()
        
        # Add bar chart for mean values
        fig.add_trace(go.Bar(
            x=grouped_df['relevance'],
            y=grouped_df['mean'],
            error_y=dict(type='data', array=grouped_df['std']),
            name='Mean Value',
            text=grouped_df['count'],
            textposition='auto',
            hovertemplate='Relevance: %{x}<br>Mean: %{y:.4f}<br>Std: %{error_y.array:.4f}<br>Count: %{text}'
        ))
        
        # Add scatter plot for median values
        fig.add_trace(go.Scatter(
            x=grouped_df['relevance'],
            y=grouped_df['median'],
            mode='markers',
            marker=dict(size=10, symbol='diamond'),
            name='Median Value'
        ))
        
        fig.update_layout(
            title=f'{feature_name} by Relevance Level ({group_name})',
            xaxis_title='Relevance Level',
            yaxis_title='Feature Value',
            xaxis=dict(tickmode='array', tickvals=list(range(5))),
            hovermode='closest',
            height=500
        )
        return fig
    else:
        plt.figure(figsize=(10, 6))
        
        # Bar chart for mean values
        plt.bar(grouped_df['relevance'], grouped_df['mean'], yerr=grouped_df['std'], 
                alpha=0.7, capsize=10, label='Mean')
        
        # Scatter plot for median values
        plt.scatter(grouped_df['relevance'], grouped_df['median'], color='red', 
                   s=100, marker='D', label='Median')
        
        # Add count as text
        for i, count in enumerate(grouped_df['count']):
            plt.text(grouped_df['relevance'].iloc[i], grouped_df['mean'].iloc[i] + grouped_df['std'].iloc[i] + 0.05, 
                    f'n={count}', ha='center')
        
        plt.title(f'{feature_name} by Relevance Level ({group_name})')
        plt.xlabel('Relevance Level')
        plt.ylabel('Feature Value')
        plt.xticks(range(5))
        plt.legend()
        plt.tight_layout()
        return plt

def plot_feature_group_importance(importance_df, use_plotly=True):
    """Plot importance by feature group.
    
    Args:
        importance_df: DataFrame with feature importance scores
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        
    Returns:
        Figure object
    """
    # Calculate sum of importance for each group
    group_importance = importance_df.groupby('group')['importance'].sum().reset_index()
    
    if use_plotly:
        fig = px.pie(
            group_importance, 
            values='importance', 
            names='group',
            title='Feature Importance by Group',
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=600)
        return fig
    else:
        plt.figure(figsize=(10, 8))
        plt.pie(group_importance['importance'], labels=group_importance['group'], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Feature Importance by Group')
        plt.axis('equal')
        plt.tight_layout()
        return plt
