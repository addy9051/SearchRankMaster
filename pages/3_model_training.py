import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

# We'll use the safe imports from model_training instead
from utils.data_loader import load_dataset, prepare_dataset_for_training, FEATURE_GROUPS
from utils.model_training import (
    train_model, 
    get_model_params, 
    get_model_approach,
    get_available_models,
    LIGHTGBM_AVAILABLE,
    XGBOOST_AVAILABLE
)
from utils.feature_analysis import calculate_feature_importance

st.set_page_config(
    page_title="Model Training - ML Search Ranking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Model Training")
st.write("Train Learning to Rank models using different approaches (pointwise, pairwise, listwise).")

# Initialize session state for trained models
if 'trained_models' not in st.session_state:
    st.session_state['trained_models'] = {}

if 'training_history' not in st.session_state:
    st.session_state['training_history'] = {}

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
        # Reset trained models if dataset changes
        st.session_state['trained_models'] = {}
        st.session_state['training_history'] = {}
    
    st.sidebar.header("Sample Size")
    train_sample_size = st.sidebar.slider(
        "Training samples:",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )
    
    val_sample_size = st.sidebar.slider(
        "Validation samples:",
        min_value=500,
        max_value=5000,
        value=1000,
        step=500
    )

# Main content
st.header("Learning to Rank Approaches")

# Display information about different approaches
approaches_tab1, approaches_tab2, approaches_tab3 = st.tabs(["Pointwise", "Pairwise", "Listwise"])

with approaches_tab1:
    st.markdown("""
    ### Pointwise Approach
    The pointwise approach treats each query-document pair independently, predicting an absolute relevance score.
    
    **Characteristics:**
    - Simplest approach to implement
    - Can use traditional regression or classification models
    - Doesn't consider the relative ordering between documents
    
    **Example Models:**
    - Linear Regression
    - Random Forest Regressor
    - LightGBM/XGBoost Regressor
    """)

with approaches_tab2:
    st.markdown("""
    ### Pairwise Approach
    The pairwise approach considers pairs of documents for a given query, learning to predict which document is more relevant.
    
    **Characteristics:**
    - Focuses on relative ordering between pairs of documents
    - Better captures the ranking nature of the problem
    - Often provides better ranking performance than pointwise
    
    **Example Models:**
    - RankNet
    - LambdaRank
    - LambdaMART (implemented in LightGBM/XGBoost)
    """)

with approaches_tab3:
    st.markdown("""
    ### Listwise Approach
    The listwise approach directly optimizes a ranking metric over the entire list of documents for a query.
    
    **Characteristics:**
    - Directly optimizes ranking metrics (NDCG, MAP)
    - Most sophisticated approach
    - Often yields the best performance but can be more complex to implement
    
    **Example Models:**
    - ListNet
    - LambdaMART with listwise loss
    - TF-Ranking with listwise loss
    """)

# Data loading section
st.header("Data Loading")

# Load data
load_data_container = st.container()

with load_data_container:
    col1, col2 = st.columns(2)
    
    with col1:
        load_data_button = st.button("Load Training & Validation Data", type="primary")
    
    with col2:
        data_loading_placeholder = st.empty()

if load_data_button or ('train_df' in st.session_state and 'val_df' in st.session_state):
    if 'train_df' not in st.session_state or 'val_df' not in st.session_state:
        data_loading_placeholder.info("Loading data...")
        
        with st.spinner(f"Loading {train_sample_size} training samples and {val_sample_size} validation samples..."):
            train_df = load_dataset(selected_dataset, "train", max_samples=train_sample_size)
            val_df = load_dataset(selected_dataset, "validation", max_samples=val_sample_size)
            
            if train_df is not None and val_df is not None:
                st.session_state['train_df'] = train_df
                st.session_state['val_df'] = val_df
                data_loading_placeholder.success("Data loaded successfully!")
            else:
                data_loading_placeholder.error("Failed to load data. Please try again.")
    else:
        train_df = st.session_state['train_df']
        val_df = st.session_state['val_df']
        data_loading_placeholder.success("Data already loaded.")
    
    # Data summary
    st.subheader("Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Samples", len(train_df))
        st.metric("Unique Training Queries", train_df['query_id'].nunique())
        
        # Training relevance distribution
        train_rel_dist = train_df['relevance'].value_counts().sort_index()
        
        fig = px.pie(
            values=train_rel_dist.values,
            names=train_rel_dist.index,
            title='Training Data: Relevance Distribution',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Validation Samples", len(val_df))
        st.metric("Unique Validation Queries", val_df['query_id'].nunique())
        
        # Validation relevance distribution
        val_rel_dist = val_df['relevance'].value_counts().sort_index()
        
        fig = px.pie(
            values=val_rel_dist.values,
            names=val_rel_dist.index,
            title='Validation Data: Relevance Distribution',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature selection
    st.header("Feature Selection")
    st.markdown("""
    Select features to use for training. Features are organized into:

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

    # Add feature group selection
    selected_groups = st.multiselect(
        "Select feature groups to use:",
        options=list(FEATURE_GROUPS.keys()),
        default=list(FEATURE_GROUPS.keys())
    )
    
    selected_features = None
    
    if selected_groups:
        selected_features = []
        for group in selected_groups:
            selected_features.extend(FEATURE_GROUPS[group])
        
        st.success(f"Selected {len(selected_features)} features from {len(selected_groups)} groups.")
    
    # Model selection and training
    st.header("Model Selection & Training")
    
    # Get available models (filtered based on installed libraries)
    available_models = get_available_models()
    
    # Display info about available libraries
    if not LIGHTGBM_AVAILABLE:
        st.warning("LightGBM is not available. Some models will not be available.")
    
    if not XGBOOST_AVAILABLE:
        st.warning("XGBoost is not available. Some models will not be available.")
    
    if not available_models:
        st.error("No models are available. Please install the required libraries.")
        st.stop()
    
    # Select approach (only show approaches that have models available)
    approach_options = list(available_models.keys())
    selected_approach = st.selectbox(
        "Select a ranking approach:",
        options=approach_options,
        index=0
    )
    
    # Select model based on approach
    selected_model = st.selectbox(
        "Select a model:",
        options=available_models[selected_approach],
        index=0
    )
    
    # Get default parameters for the selected model
    default_params = get_model_params(selected_model)
    
    # Display and allow editing of model parameters
    st.subheader("Model Parameters")
    
    model_params = {}
    
    # Create parameter inputs based on model type
    if selected_model == 'Linear Regression (Ridge)':
        col1, col2 = st.columns(2)
        with col1:
            model_params['alpha'] = st.number_input("Alpha (regularization)", 
                                                  min_value=0.0, max_value=10.0, 
                                                  value=default_params.get('alpha', 1.0), step=0.1)
        with col2:
            model_params['max_iter'] = st.number_input("Max Iterations", 
                                                     min_value=100, max_value=10000, 
                                                     value=default_params.get('max_iter', 1000), step=100)
    
    elif selected_model == 'Random Forest Regressor':
        col1, col2 = st.columns(2)
        with col1:
            model_params['n_estimators'] = st.number_input("Number of Estimators", 
                                                         min_value=10, max_value=500, 
                                                         value=default_params.get('n_estimators', 100), step=10)
            model_params['min_samples_split'] = st.number_input("Min Samples Split", 
                                                              min_value=2, max_value=20, 
                                                              value=default_params.get('min_samples_split', 2), step=1)
        with col2:
            model_params['max_depth'] = st.number_input("Max Depth", 
                                                      min_value=5, max_value=100, 
                                                      value=default_params.get('max_depth', 10), step=5)
            model_params['min_samples_leaf'] = st.number_input("Min Samples Leaf", 
                                                             min_value=1, max_value=20, 
                                                             value=default_params.get('min_samples_leaf', 1), step=1)
    
    elif selected_model.startswith('LightGBM'):
        col1, col2 = st.columns(2)
        with col1:
            model_params['num_leaves'] = st.number_input("Number of Leaves", 
                                                      min_value=10, max_value=255, 
                                                      value=default_params.get('num_leaves', 31), step=5)
            model_params['learning_rate'] = st.number_input("Learning Rate", 
                                                          min_value=0.01, max_value=0.5, 
                                                          value=default_params.get('learning_rate', 0.1), step=0.01,
                                                          format="%.2f")
        with col2:
            model_params['n_estimators'] = st.number_input("Number of Estimators", 
                                                         min_value=10, max_value=500, 
                                                         value=default_params.get('n_estimators', 100), step=10)
            model_params['max_depth'] = st.number_input("Max Depth", 
                                                      min_value=-1, max_value=100, 
                                                      value=default_params.get('max_depth', -1), step=5)
        
        # Additional parameters for LambdaMART
        if 'Ranker' in selected_model:
            st.markdown("### LambdaMART Parameters")
            model_params['ndcg_eval_at'] = [1, 3, 5, 10]  # Fixed evaluation points for simplicity
    
    elif selected_model.startswith('XGBoost'):
        col1, col2 = st.columns(2)
        with col1:
            model_params['max_depth'] = st.number_input("Max Depth", 
                                                      min_value=3, max_value=20, 
                                                      value=default_params.get('max_depth', 6), step=1)
            model_params['learning_rate'] = st.number_input("Learning Rate", 
                                                          min_value=0.01, max_value=0.5, 
                                                          value=default_params.get('learning_rate', 0.1), step=0.01,
                                                          format="%.2f")
        with col2:
            model_params['n_estimators'] = st.number_input("Number of Estimators", 
                                                         min_value=10, max_value=500, 
                                                         value=default_params.get('n_estimators', 100), step=10)
    
    elif selected_model.startswith('TF-Ranking'):
        col1, col2 = st.columns(2)
        with col1:
            model_params['learning_rate'] = st.number_input("Learning Rate", 
                                                          min_value=0.001, max_value=0.1, 
                                                          value=default_params.get('learning_rate', 0.05), step=0.001,
                                                          format="%.3f")
            model_params['batch_size'] = st.number_input("Batch Size", 
                                                       min_value=32, max_value=512, 
                                                       value=default_params.get('batch_size', 128), step=32)
        with col2:
            model_params['epochs'] = st.number_input("Epochs", 
                                                   min_value=5, max_value=50, 
                                                   value=default_params.get('epochs', 10), step=1)
            
            # Loss function selection
            if selected_model == 'TF-Ranking (pairwise)':
                loss_options = ["pairwise_logistic_loss", "pairwise_hinge_loss"]
            else:  # listwise
                loss_options = ["softmax_loss", "list_mle_loss"]
            
            model_params['loss'] = st.selectbox("Loss Function", options=loss_options)
        
        # Hidden layers specification
        st.markdown("### Model Architecture")
        hidden_layers_str = st.text_input("Hidden Layers (comma-separated)", 
                                       value=",".join(map(str, default_params.get('hidden_layers', [64, 32]))))
        model_params['hidden_layers'] = [int(x.strip()) for x in hidden_layers_str.split(",") if x.strip()]
    
    # Train model button
    train_button = st.button("Train Model", type="primary", key="train_model_button")
    
    if train_button:
        if 'train_df' in st.session_state and 'val_df' in st.session_state:
            # Create a progress bar
            progress_bar = st.progress(0, text="Initializing training...")
            
            # Create a unique model ID
            model_id = f"{selected_model}_{int(time.time())}"
            
            # Train the model
            with st.spinner(f"Training {selected_model}..."):
                try:
                    model, history = train_model(
                        st.session_state['train_df'],
                        st.session_state['val_df'],
                        selected_model,
                        model_params,
                        features_to_use=selected_features,
                        progress_bar=progress_bar
                    )
                    
                    if model is not None:
                        # Store the trained model in session state
                        st.session_state['trained_models'][model_id] = {
                            'model': model,
                            'name': selected_model,
                            'approach': selected_approach,
                            'params': model_params,
                            'features': selected_features,
                            'dataset': selected_dataset
                        }
                        
                        # Store training history
                        st.session_state['training_history'][model_id] = history
                        
                        st.success(f"Successfully trained {selected_model}!")
                        
                        # Display training history
                        st.subheader("Training History")
                        
                        # Plot training metrics
                        if 'train_loss' in history and history['train_loss']:
                            fig = go.Figure()
                            
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
                            
                            fig.update_layout(
                                title='Training and Validation Loss',
                                xaxis_title='Iteration',
                                yaxis_title='Loss',
                                height=400,
                                margin=dict(l=40, r=40, t=60, b=40)  # Add some margin
                            )
                            st.plotly_chart(fig, use_container_width=True, 
                                         key=f"training_loss_{model_id}_{int(time.time())}")
                        
                        # Plot additional metrics if available (e.g., NDCG for ranking models)
                        if 'train_metrics' in history and 'val_metrics' in history:
                            # Find common metrics
                            train_metrics = history['train_metrics']
                            val_metrics = history['val_metrics']
                            
                            for metric_name in train_metrics:
                                if metric_name in val_metrics:
                                    # Check if metric is a list (e.g., ndcg@1, ndcg@5)
                                    if isinstance(train_metrics[metric_name], list) and len(train_metrics[metric_name]) > 1:
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Scatter(
                                            y=train_metrics[metric_name],
                                            mode='lines',
                                            name=f'Training {metric_name}'
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            y=val_metrics[metric_name],
                                            mode='lines',
                                            name=f'Validation {metric_name}'
                                        ))
                                        
                                        fig.update_layout(
                                            title=f'Training and Validation {metric_name.upper()}',
                                            xaxis_title='Iteration',
                                            yaxis_title=metric_name.upper(),
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True, key=f"metric_chart_{metric_name}_{model_id}")
                    else:
                        st.error("Failed to train model. Please check parameters and try again.")
                
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.error("Please load the data first.")

    # Trained Models Section
    st.header("Trained Models")
    
    if 'trained_models' in st.session_state and st.session_state['trained_models']:
        # Display a table of trained models
        models_data = []
        
        for model_id, model_info in st.session_state['trained_models'].items():
            models_data.append({
                'ID': model_id,
                'Name': model_info['name'],
                'Approach': model_info['approach'],
                'Dataset': model_info['dataset'],
                'Features': 'All' if model_info['features'] is None else f"{len(model_info['features'])} selected"
            })
        
        models_df = pd.DataFrame(models_data)
        st.dataframe(models_df)
        
        # Select a model to view details
        if models_df.empty:
            st.info("No trained models available.")
        else:
            selected_model_id = st.selectbox(
                "Select a model to view details:",
                options=list(st.session_state['trained_models'].keys()),
                format_func=lambda x: f"{st.session_state['trained_models'][x]['name']} ({x.split('_')[-1]})"
            )
            
            if selected_model_id:
                model_info = st.session_state['trained_models'][selected_model_id]
                history = st.session_state['training_history'].get(selected_model_id, {})
                
                st.subheader(f"Model Details: {model_info['name']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Approach**: {model_info['approach']}")
                    st.markdown(f"**Dataset**: {model_info['dataset']}")
                    
                    # Display model parameters
                    st.markdown("**Parameters:**")
                    for param, value in model_info['params'].items():
                        st.markdown(f"- {param}: {value}")
                
                with col2:
                    # Display feature information
                    if model_info['features'] is None:
                        st.markdown("**Features**: All 136 features used")
                    else:
                        st.markdown(f"**Features**: {len(model_info['features'])} features selected")
                        
                        # Group features by their groups
                        feature_groups = {}
                        for feat in model_info['features']:
                            group = None
                            for group_name, indices in FEATURE_GROUPS.items():
                                if feat in indices:
                                    group = group_name
                                    break
                            
                            if group not in feature_groups:
                                feature_groups[group] = []
                            
                            feature_groups[group].append(feat)
                        
                        # Display features by group
                        for group, feats in feature_groups.items():
                            st.markdown(f"- {group}: {len(feats)} features")
                
                # Display training history if available
                if history:
                    st.subheader("Training History")
                    
                    # Plot training metrics
                    if 'train_loss' in history and history['train_loss']:
                        fig = go.Figure()
                        
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
                        
                        fig.update_layout(
                            title='Training and Validation Loss',
                            xaxis_title='Iteration',
                            yaxis_title='Loss',
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)  # Add some margin
                        )
                        st.plotly_chart(fig, use_container_width=True, 
                                     key=f"model_details_loss_{selected_model_id}_{int(time.time())}")
                
                # Plot additional metrics if available
                if 'train_metrics' in history and 'val_metrics' in history:
                    train_metrics = history['train_metrics']
                    val_metrics = history['val_metrics']
                    
                    for metric_name in train_metrics:
                        if metric_name in val_metrics:
                            if isinstance(train_metrics[metric_name], list) and len(train_metrics[metric_name]) > 1:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    y=train_metrics[metric_name],
                                    mode='lines',
                                    name=f'Training {metric_name}'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    y=val_metrics[metric_name],
                                    mode='lines',
                                    name=f'Validation {metric_name}'
                                ))
                                
                                fig.update_layout(
                                    title=f'Training and Validation {metric_name.upper()}',
                                    xaxis_title='Iteration',
                                    yaxis_title=metric_name.upper(),
                                    height=400,
                                    margin=dict(l=40, r=40, t=60, b=40)  # Add some margin
                                )
                                st.plotly_chart(fig, use_container_width=True, 
                                            key=f"model_details_metric_{metric_name}_{selected_model_id}_{int(time.time())}")
        
        # Clear button to remove the selected model
        if st.button("Delete this model", key=f"delete_model_button_{selected_model_id}"):
            if selected_model_id in st.session_state['trained_models']:
                del st.session_state['trained_models'][selected_model_id]
            
            if selected_model_id in st.session_state['training_history']:
                del st.session_state['training_history'][selected_model_id]
            
            st.success(f"Model {selected_model_id} deleted.")
            st.rerun()
    else:
        st.info("No models have been trained yet. Use the form above to train a model.")
else:
    st.info("Please load the training and validation data to begin model training.")
