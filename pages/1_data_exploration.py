import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_dataset, load_dataset_info, get_dataset_stats, get_feature_names, get_feature_group, FEATURE_GROUPS

st.set_page_config(
    page_title="Data Exploration - ML Search Ranking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Data Exploration")
st.write("Explore the MSLR datasets to understand their structure, feature distributions, and relevance judgments.")

# Select dataset
dataset_options = {
    "mslr_web10k": "MSLR-WEB10K",
    "mslr_web30k": "MSLR-WEB30K"
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
    
    # Get dataset stats to determine the maximum sample size
    dataset_stats = get_dataset_stats(selected_dataset, selected_split, 1000)  # Load minimal sample to get stats
    max_samples = dataset_stats.get('total_rows', 10000) if dataset_stats else 10000
    
    # Make sure max_samples is at least 100 and not None
    if max_samples is None or max_samples < 100:
        max_samples = 10000
    
    st.sidebar.header("Sample Size")
    sample_size = st.sidebar.slider(
        "Number of samples to load:",
        min_value=100,
        max_value=min(max_samples, 100000),  # Cap at 100k to avoid UI issues
        value=min(1000, max_samples),  # Default to 1000 or max_samples if smaller
        step=100,
        help=f"Total available rows: {max_samples}"
    )
    
    st.sidebar.header("Visualization Settings")
    show_correlations = st.sidebar.checkbox("Show Feature Correlations", value=True)
    show_distributions = st.sidebar.checkbox("Show Feature Distributions", value=True)
    show_query_analysis = st.sidebar.checkbox("Show Query Analysis", value=True)

# Load dataset information
try:
    dataset_info = load_dataset_info(selected_dataset)
    
    # Display dataset overview
    st.header("Dataset Overview")
    if dataset_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{dataset_options[selected_dataset]} Dataset")
            st.markdown(f"**Description**: {dataset_info.get('description', 'No description available')}")
            st.markdown("""
            **Feature Organization**:
            - 136 features per query-document pair
            - Features 1-125: Document field features (body, anchor, title, URL, whole doc)
            - Features 126-136: Miscellaneous features (URL structure, links, ranking, quality, clicks)
            """)
            st.markdown(f"**Relevance Scale**: 0-4 (0: irrelevant, 4: perfectly relevant)")
            
            # Display dataset structure
            if st.button("Show Dataset Structure"):
                st.json(dataset_info.get('features', {}))
        
        with col2:
            # Get dataset statistics
            st.subheader("Dataset Statistics")
            with st.spinner("Calculating dataset statistics..."):
                stats = get_dataset_stats(selected_dataset, selected_split, sample_size)
                
                if stats:
                    st.markdown(f"**Total Records**: {stats.get('total_records', 'N/A')}")
                    st.markdown(f"**Unique Queries (sample)**: {stats.get('unique_queries_sample', 'N/A')}")
                    st.markdown(f"**Avg. Documents per Query**: {stats.get('docs_per_query_avg', 0)}")
                    
                    # Display relevance distribution
                    st.subheader("Relevance Distribution")
                    if 'relevance_percentage' in stats:
                        rel_dist = pd.DataFrame({
                            'Relevance': list(stats['relevance_percentage'].keys()),
                            'Percentage': list(stats['relevance_percentage'].values())
                        }).sort_values('Relevance')
                        
                        fig = px.bar(
                            rel_dist, 
                            x='Relevance', 
                            y='Percentage',
                            title='Relevance Distribution (%)',
                            labels={'Relevance': 'Relevance Label', 'Percentage': 'Percentage (%)'},
                            color='Relevance',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not calculate relevance distribution.")
    else:
        st.error("Failed to load dataset information.")
except Exception as e:
    st.error(f"Error loading dataset information: {str(e)}")

# Load dataset
with st.spinner(f"Loading {selected_dataset} {selected_split} data..."):
    try:
        df = load_dataset(selected_dataset, selected_split, max_samples=sample_size)
        if df is None:
            st.error("Failed to load dataset. Please check if the dataset files exist and are in the correct format.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

# Display sample data
st.header(f"Data Sample ({selected_split.capitalize()} Split)")
st.dataframe(df.head(10))

# Basic statistics
st.header("Basic Statistics")
    
col1, col2 = st.columns(2)
    
with col1:
    # Query statistics
    unique_queries = df['query_id'].nunique()
    st.metric("Unique Queries", unique_queries)
    
    # Documents per query distribution
    docs_per_query = df.groupby('query_id').size().describe()
    st.write("Documents per Query Distribution:")
    st.write(docs_per_query)
    
    # Documents per query histogram
    fig = px.histogram(
        df.groupby('query_id').size().reset_index(name='count'),
        x='count',
        title='Distribution of Documents per Query',
        nbins=30,
        color_discrete_sequence=['teal']
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Relevance distribution
    relevance_counts = df['relevance'].value_counts().sort_index()
    
    fig = px.pie(
        values=relevance_counts.values,
        names=relevance_counts.index,
        title='Relevance Distribution',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Relevance distribution by query
    avg_relevance = df.groupby('query_id')['relevance'].mean().reset_index()
    fig = px.histogram(
        avg_relevance,
        x='relevance',
        title='Distribution of Average Relevance per Query',
        nbins=20,
        color_discrete_sequence=['purple']
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Feature exploration
st.header("Feature Exploration")

# Feature group selection
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

# Get all feature columns (handle both 'feature_X' and 'fX' formats)
feature_columns = []
feature_number_to_column = {}

for col in df.columns:
    # Check for 'feature_X' format
    if col.startswith('feature_') and col.split('_')[-1].isdigit():
        feature_num = int(col.split('_')[-1])
        feature_columns.append(col)
        feature_number_to_column[feature_num] = col
    # Check for 'fX' format
    elif col.startswith('f') and col[1:].isdigit():
        feature_num = int(col[1:])
        feature_columns.append(col)
        feature_number_to_column[feature_num] = col

# Feature selection for visualization
st.subheader("Explore Feature Distributions")

# Feature group filter
selected_group = st.selectbox(
    "Filter by feature group:",
    options=list(FEATURE_GROUPS.keys()) + ["All Features"],
    index=len(FEATURE_GROUPS)
)

# Get feature options based on the selected group
if selected_group == "All Features":
    # For 'All Features', use all available feature numbers
    feature_options = sorted(list(feature_number_to_column.keys()))
else:
    # For specific groups, only include features that exist in both the group and our column mapping
    feature_options = [f for f in FEATURE_GROUPS[selected_group] if f in feature_number_to_column]

# Debug: Print selected feature options
st.write(f"### Selected Group: {selected_group}")
st.write(f"Feature options: {feature_options}")

if not feature_options:
    st.warning(f"No features available in the selected group: {selected_group}")
    st.warning(f"Feature columns in dataset: {feature_columns}")
    selected_feature = None
else:
    selected_feature = st.selectbox(
        "Select a feature to visualize:",
        options=feature_options,
        format_func=lambda x: f"Feature {x} ({get_feature_group(x)})" if x is not None else "No features available"
    )

# Feature visualization
if selected_feature is not None:
    feature_col = f"feature_{selected_feature}"
    
    # Ensure the feature column exists in the DataFrame
    if feature_col not in df.columns:
        st.error(f"Feature column '{feature_col}' not found in the dataset")
        st.stop()
    
    # Display basic statistics for the selected feature
    feature_stats = df[feature_col].describe()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Feature {selected_feature} Statistics")
        st.write(feature_stats)
        
        # Add normality test (only if we have enough samples)
        feature_data = df[feature_col].dropna()
        if len(feature_data) > 7:  # normaltest requires at least 8 samples
            try:
                data_values = feature_data.values
                
                # Check for any non-finite values
                if not np.all(np.isfinite(data_values)):
                    data_values = data_values[np.isfinite(data_values)]
                
                if len(data_values) >= 8:
                    try:
                        stat, p_value = scipy_stats.normaltest(data_values, axis=0)
                    except TypeError:
                        stat, p_value = scipy_stats.normaltest(data_values)
                        
                    st.write(f"Normality Test p-value: {p_value:.4f}")
                    st.write("Feature is " + ("normally distributed" if p_value > 0.05 else "not normally distributed"))
                
            except Exception:
                pass  # Silently handle any errors in the test
        else:
            st.warning("Not enough samples for normality test")
    
    with col2:
        # Correlation with relevance
        corr = df[[feature_col, 'relevance']].corr().iloc[0, 1]
        st.subheader("Correlation with Relevance")
        st.metric("Pearson Correlation", f"{corr:.4f}")
        
        # Add correlation interpretation
        if abs(corr) < 0.1:
            st.write("Very weak correlation")
        elif abs(corr) < 0.3:
            st.write("Weak correlation")
        elif abs(corr) < 0.5:
            st.write("Moderate correlation")
        elif abs(corr) < 0.7:
            st.write("Strong correlation")
        else:
            st.write("Very strong correlation")
        
        # Distribution plot
        st.subheader(f"Distribution of Feature {selected_feature}")
        
        fig = px.histogram(
            df, 
            x=feature_col,
            marginal="box",
            title=f"Distribution of Feature {selected_feature}",
            color_discrete_sequence=['teal'],
            nbins=50
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature by relevance
        st.subheader(f"Feature {selected_feature} by Relevance Label")
        
        # Boxplot by relevance
        fig = px.box(
            df, 
            x="relevance", 
            y=feature_col,
            title=f"Feature {selected_feature} Distribution by Relevance Label",
            color="relevance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add violin plot
        fig = px.violin(
            df,
            x="relevance",
            y=feature_col,
            title=f"Feature {selected_feature} Distribution by Relevance Label (Violin Plot)",
            color="relevance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlations
    if show_correlations:
        st.header("Feature Correlations")
        
        # Select features to show correlations for
        st.subheader("Feature Correlation Analysis")
        
        # Get top correlated features with relevance
        feature_cols = [f'feature_{i}' for i in range(1,137)]
        correlations = df[feature_cols + ['relevance']].corr()['relevance'].sort_values(ascending=False)
        
        # Plot top correlations
        fig = px.bar(
            x=correlations.index[1:][:20],  # Exclude relevance itself and show top 20
            y=correlations.values[1:][:20],
            title='Top 20 Features Correlated with Relevance',
            labels={'x': 'Feature', 'y': 'Correlation'},
            color=correlations.values[1:][:20],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix for selected feature group
        if selected_group != "All Features":
            st.subheader(f"Correlation Matrix for {selected_group}")
            group_features = [f'feature_{i}' for i in FEATURE_GROUPS[selected_group]]
            corr_matrix = df[group_features + ['relevance']].corr()
            
            fig = px.imshow(
                corr_matrix,
                title=f"Correlation Matrix for {selected_group}",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
# Query exploration
if show_query_analysis:
    st.header("Query Analysis")
    
    # Get unique query IDs
    unique_qids = df['query_id'].unique()
    
    selected_qid = st.selectbox(
        "Select a query to explore:",
        options=unique_qids,
        index=0
    )
    
    if selected_qid:
        # Filter data for the selected query
        query_df = df[df['query_id'] == selected_qid]
        
        st.subheader(f"Documents for Query {selected_qid}")
        st.write(f"Number of documents: {len(query_df)}")
        
        # Relevance distribution for this query
        rel_counts = query_df['relevance'].value_counts().sort_index()
        
        fig = px.bar(
            x=rel_counts.index,
            y=rel_counts.values,
            title=f"Relevance Distribution for Query {selected_qid}",
            labels={'x': 'Relevance Label', 'y': 'Count'},
            color=rel_counts.index,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions for this query
        st.subheader(f"Feature Distributions for Query {selected_qid}")
        
        # Select features to compare
        selected_features = st.multiselect(
            "Select features to compare:",
            options=[f'feature_{i}' for i in range(1,137)],
            default=[f'feature_{i}' for i in range(1,6)]  # Default to first 5 features
        )
        
        if selected_features:
            # Create box plots for selected features
            fig = go.Figure()
            for feature in selected_features:
                fig.add_trace(go.Box(
                    y=query_df[feature],
                    name=feature
                ))
            
            # Update layout for the box plot
            fig.update_layout(
                title=f'Feature Distributions for Query {selected_qid}',
                xaxis_title='Feature',
                yaxis_title='Value',
                showlegend=True
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)