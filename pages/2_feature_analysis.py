import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from utils.data_loader import load_dataset, get_feature_names, get_feature_group, FEATURE_GROUPS
from utils.feature_analysis import (
    calculate_feature_importance, 
    calculate_feature_correlations,
    analyze_feature_distributions,
    analyze_feature_by_relevance,
    plot_feature_importance,
    plot_feature_correlation_heatmap,
    plot_feature_distribution,
    plot_feature_by_relevance,
    plot_feature_group_importance
)

st.set_page_config(
    page_title="Feature Analysis - ML Search Ranking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Feature Analysis")
st.write("Analyze features to understand their importance, correlations, and relationships with relevance judgments.")

# Select dataset
dataset_options = {
    "mslr_web10k": "MSLR-WEB10K"
}

# Check if dataset is already selected in session state
if 'selected_dataset' not in st.session_state:
    st.session_state['selected_dataset'] = 'mslr_web10k'

# Dataset selection in sidebar
with st.sidebar:
    st.header("Dataset Selection")
    selected_dataset = st.selectbox(
        "Choose a dataset:",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        index=list(dataset_options.keys()).index(st.session_state['selected_dataset'])
    )
    
    # Update session state if changed
    if selected_dataset != st.session_state['selected_dataset']:
        st.session_state['selected_dataset'] = selected_dataset
        
    st.sidebar.header("Data Split Selection")
    selected_split = st.sidebar.selectbox(
        "Choose a data split:",
        options=["train", "validation", "test"],
        index=0
    )
    
    st.sidebar.header("Sample Size")
    sample_size = st.sidebar.slider(
        "Number of samples to analyze:",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )

# Load data
st.header("Loading Data")
with st.spinner(f"Loading {sample_size} samples from {dataset_options[selected_dataset]} {selected_split} split..."):
    df = load_dataset(selected_dataset, selected_split, max_samples=sample_size)

if df is not None:
    st.success(f"Loaded {len(df)} samples from {dataset_options[selected_dataset]} {selected_split} split")
    
    # Feature importance analysis
    st.header("Feature Importance Analysis")
    
    importance_method = st.radio(
        "Select importance calculation method:",
        options=["mutual_info", "correlation"],
        format_func=lambda x: "Mutual Information" if x == "mutual_info" else "Spearman Correlation",
        horizontal=True
    )
    
    with st.spinner("Calculating feature importance..."):
        importance_df = calculate_feature_importance(df, method=importance_method)
    
    # Display feature importance
    st.subheader("Top Features by Importance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_feature_importance(importance_df, top_n=20, use_plotly=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            importance_df.head(20).reset_index(drop=True),
            column_config={
                "feature_index": st.column_config.NumberColumn("Feature", format="%d"),
                "importance": st.column_config.NumberColumn("Importance", format="%.4f"),
                "group": st.column_config.TextColumn("Feature Group")
            },
            height=600
        )
    
    # Feature importance by group
    st.subheader("Feature Importance by Group")
    
    fig = plot_feature_group_importance(importance_df, use_plotly=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation analysis
    st.header("Feature Correlation Analysis")
    
    top_n_corr = st.slider(
        "Number of top features to include in correlation analysis:",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )
    
    with st.spinner("Calculating feature correlations..."):
        corr_matrix = calculate_feature_correlations(df, top_n=top_n_corr)
    
    # Display correlation matrix
    st.subheader(f"Correlation Matrix for Top {top_n_corr} Features")
    
    fig = plot_feature_correlation_heatmap(corr_matrix, use_plotly=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual feature analysis
    st.header("Individual Feature Analysis")
    
    # Feature group filter
    st.subheader("Feature Groups")
    st.markdown("""
    The features are organized into two main categories:

    1. **Document Field Features** (1-125):
       - Body text features (1-25)
       - Anchor text features (26-50)
       - Title features (51-75)
       - URL features (76-100)
       - Whole document features (101-125)

    2. **Miscellaneous Features** (126-136):
       - URL structure features (126-127)
       - Link features (128-129)
       - Ranking features (130-131)
       - Quality features (132-133)
       - Click-based features (134-136)
    """)

    selected_group = st.selectbox(
        "Filter by feature group:",
        options=list(FEATURE_GROUPS.keys()) + ["All Features"],
        index=len(FEATURE_GROUPS),
        help="Select a feature group to analyze its features"
    )
    
    if selected_group == "All Features":
        feature_options = list(range(1,137))
    else:
        feature_options = FEATURE_GROUPS[selected_group]
    
    selected_feature = st.selectbox(
        "Select a feature to analyze:",
        options=feature_options,
        format_func=lambda x: f"Feature {x} ({get_feature_group(x)})"
    )
    
    if selected_feature is not None:
        # Feature distribution
        st.subheader(f"Distribution Analysis for Feature {selected_feature}")
        
        feature_col = f"feature_{selected_feature}"
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic statistics
            feature_stats = df[feature_col].describe()
            st.write(feature_stats)
        
        with col2:
            # Correlation with relevance
            corr = df[[feature_col, 'relevance']].corr().iloc[0, 1]
            st.metric("Correlation with Relevance", f"{corr:.4f}")
            
            # Add more stats as needed
            st.write(f"Unique values: {df[feature_col].nunique()}")
            st.write(f"Missing values: {df[feature_col].isna().sum()}")
        
        # Distribution plot
        fig = plot_feature_distribution(df, selected_feature, use_plotly=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature by relevance
        st.subheader(f"Feature {selected_feature} vs. Relevance")
        
        # Get feature statistics by relevance
        with st.spinner("Analyzing feature by relevance..."):
            relevance_stats = analyze_feature_by_relevance(df, selected_feature)
        
        # Display statistics
        st.write("Feature statistics by relevance level:")
        st.dataframe(
            relevance_stats,
            column_config={
                "relevance": st.column_config.NumberColumn("Relevance", format="%d"),
                "mean": st.column_config.NumberColumn("Mean", format="%.4f"),
                "median": st.column_config.NumberColumn("Median", format="%.4f"),
                "std": st.column_config.NumberColumn("Std Dev", format="%.4f"),
                "count": st.column_config.NumberColumn("Count", format="%d")
            }
        )
        
        # Plot feature by relevance
        fig = plot_feature_by_relevance(relevance_stats, selected_feature, use_plotly=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Set Analysis
    st.header("Feature Set Analysis")
    
    # Select multiple features for comparison
    st.subheader("Compare Multiple Features")
    
    # Default to top 5 features
    default_features = importance_df.head(5)['feature_index'].tolist()
    
    multi_feature_options = st.multiselect(
        "Select features to compare:",
        options=list(range(1,137)),
        default=default_features,
        format_func=lambda x: f"Feature {x} ({get_feature_group(x)})"
    )
    
    if multi_feature_options:
        # Create feature distribution plot for selected features
        st.subheader("Box Plot Comparison")
        
        # Prepare data for plotting
        compare_data = []
        
        for feat_idx in multi_feature_options:
            feat_data = df[f'feature_{feat_idx}'].tolist()
            feat_name = f'Feature {feat_idx}'
            
            for val in feat_data:
                compare_data.append({
                    'Feature': feat_name,
                    'Value': val
                })
        
        compare_df = pd.DataFrame(compare_data)
        
        # Create box plot
        fig = px.box(
            compare_df,
            x='Feature',
            y='Value',
            color='Feature',
            title=f'Distribution Comparison for Selected Features',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation matrix
        st.subheader("Correlation Matrix for Selected Features")
        
        # Calculate correlation matrix
        feature_cols = [f'feature_{i}' for i in multi_feature_options]
        feature_names = [f'Feature {i}' for i in multi_feature_options]
        
        selected_corr = df[feature_cols].corr()
        selected_corr.columns = feature_names
        selected_corr.index = feature_names
        
        # Display correlation heatmap
        fig = px.imshow(
            selected_corr,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Correlation Matrix for Selected Features'
        )
        fig.update_layout(height=600, width=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature vs. Relevance Analysis
        st.subheader("Feature Impact on Relevance")
        
        for feat_idx in multi_feature_options:
            feature_col = f'feature_{feat_idx}'
            
            # Calculate mean value by relevance
            mean_by_rel = df.groupby('relevance')[feature_col].mean().reset_index()
            
            # Add to the plot
            fig = px.line(
                mean_by_rel,
                x='relevance',
                y=feature_col,
                markers=True,
                title=f'Mean Value of Feature {feat_idx} by Relevance Level',
                labels={
                    'relevance': 'Relevance Label',
                    feature_col: f'Mean Value of Feature {feat_idx}'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Failed to load dataset. Please try again with a different dataset or split.")
