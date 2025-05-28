import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import model libraries with error handling
from utils.model_training import LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE

from utils.data_loader import load_dataset, FEATURE_GROUPS
from utils.evaluation import (
    evaluate_model_by_query,
    get_feature_importance,
    plot_metric_distribution,
    plot_metrics_comparison,
    plot_feature_importance_comparison,
    get_predictions_by_query,
    plot_training_history
)

st.set_page_config(
    page_title="Model Evaluation - ML Search Ranking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Model Evaluation")
st.write("Evaluate trained models and compare their performance on test data.")

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
    
    st.sidebar.header("Test Data Size")
    test_sample_size = st.sidebar.slider(
        "Test samples:",
        min_value=1000,
        max_value=10000,
        value=2000,
        step=1000
    )

# Check if any models have been trained
if 'trained_models' not in st.session_state or not st.session_state['trained_models']:
    st.warning("No trained models found. Please go to the 'Model Training' page to train models first.")
    st.stop()

# Filter models by the selected dataset
dataset_models = {
    model_id: model_info 
    for model_id, model_info in st.session_state['trained_models'].items()
    if model_info['dataset'] == selected_dataset
}

if not dataset_models:
    st.warning(f"No trained models found for the {dataset_options[selected_dataset]} dataset. Please train models on this dataset first.")
    st.stop()

# Load test data button
test_data_container = st.container()

with test_data_container:
    col1, col2 = st.columns(2)
    
    with col1:
        load_test_button = st.button("Load Test Data", type="primary")
    
    with col2:
        test_loading_placeholder = st.empty()

# Load test data if button is clicked or if already loaded
if load_test_button or 'test_df' in st.session_state:
    if 'test_df' not in st.session_state:
        test_loading_placeholder.info("Loading test data...")
        
        with st.spinner(f"Loading {test_sample_size} test samples..."):
            test_df = load_dataset(selected_dataset, "test", max_samples=test_sample_size)
            
            if test_df is not None:
                st.session_state['test_df'] = test_df
                test_loading_placeholder.success("Test data loaded successfully!")
            else:
                test_loading_placeholder.error("Failed to load test data. Please try again.")
    else:
        test_df = st.session_state['test_df']
        test_loading_placeholder.success("Test data already loaded.")
    
    # Display test data summary
    st.subheader("Test Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Test Samples", len(test_df))
        st.metric("Unique Test Queries", test_df['query_id'].nunique())
    
    with col2:
        # Test relevance distribution
        test_rel_dist = test_df['relevance'].value_counts().sort_index()
        
        fig = px.pie(
            values=test_rel_dist.values,
            names=test_rel_dist.index,
            title='Test Data: Relevance Distribution',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Evaluation Section
    st.header("Model Evaluation")
    
    # Select models to evaluate
    selected_model_ids = st.multiselect(
        "Select models to evaluate:",
        options=list(dataset_models.keys()),
        default=[list(dataset_models.keys())[0]] if dataset_models else [],
        format_func=lambda x: f"{dataset_models[x]['name']} ({x.split('_')[-1]})"
    )
    
    if selected_model_ids:
        evaluate_button = st.button("Evaluate Selected Models", type="primary")
        
        if evaluate_button:
            # Initialize containers for results
            if 'evaluation_results' not in st.session_state:
                st.session_state['evaluation_results'] = {}
            
            # Evaluate each selected model
            for model_id in selected_model_ids:
                if model_id not in st.session_state['evaluation_results']:
                    model_info = dataset_models[model_id]
                    model = model_info['model']
                    model_name = model_info['name']
                    selected_features = model_info['features']
                    
                    with st.spinner(f"Evaluating {model_name} (ID: {model_id})..."):
                        try:
                            # Evaluate model
                            results_df, overall_metrics = evaluate_model_by_query(
                                model, 
                                st.session_state['test_df'], 
                                model_name,
                                features_to_use=selected_features
                            )
                            
                            # Get feature importance
                            importance_df = get_feature_importance(
                                model, 
                                model_name, 
                                feature_count=136,
                                features_to_use=selected_features
                            )
                            
                            # Store results
                            st.session_state['evaluation_results'][model_id] = {
                                'results_df': results_df,
                                'overall_metrics': overall_metrics,
                                'importance_df': importance_df
                            }
                            
                            st.success(f"Evaluated {model_name} (ID: {model_id})")
                        
                        except Exception as e:
                            st.error(f"Error evaluating {model_name} (ID: {model_id}): {str(e)}")
            
            # Display evaluation results
            st.subheader("Evaluation Results")
            
            # Check if we have results to display
            evaluated_models = [
                model_id for model_id in selected_model_ids 
                if model_id in st.session_state['evaluation_results']
            ]
            
            if evaluated_models:
                # Metrics comparison
                metrics_list = [
                    st.session_state['evaluation_results'][model_id]['overall_metrics']
                    for model_id in evaluated_models
                ]
                
                model_names = [
                    dataset_models[model_id]['name']
                    for model_id in evaluated_models
                ]
                
                # Plot metrics comparison
                st.markdown("### Metrics Comparison")
                
                fig = plot_metrics_comparison(metrics_list, model_names, use_plotly=True)
                st.plotly_chart(fig, use_container_width=True, key=f"metrics_comparison_{'_'.join(evaluated_models)}")
                
                # Display detailed metrics table
                st.markdown("### Detailed Metrics")
                
                metrics_data = []
                for i, model_id in enumerate(evaluated_models):
                    metrics = st.session_state['evaluation_results'][model_id]['overall_metrics']
                    model_name = dataset_models[model_id]['name']
                    
                    row = {'Model': model_name}
                    row.update(metrics)
                    metrics_data.append(row)
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df)
                
                # Feature importance comparison
                if len(evaluated_models) > 1:
                    st.markdown("### Feature Importance Comparison")
                    
                    importance_dfs = [
                        st.session_state['evaluation_results'][model_id]['importance_df']
                        for model_id in evaluated_models
                    ]
                    
                    fig = plot_feature_importance_comparison(
                        importance_dfs, 
                        model_names, 
                        top_n=10,
                        use_plotly=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_{'_'.join(evaluated_models)}")
                
                # Individual model details
                st.markdown("### Individual Model Performance")
                
                # Select a model to view detailed performance
                selected_detail_model = st.selectbox(
                    "Select a model to view detailed performance:",
                    options=evaluated_models,
                    format_func=lambda x: f"{dataset_models[x]['name']} ({x.split('_')[-1]})"
                )
                
                if selected_detail_model:
                    model_info = dataset_models[selected_detail_model]
                    eval_results = st.session_state['evaluation_results'][selected_detail_model]
                    
                    st.subheader(f"Detailed Performance: {model_info['name']}")
                    
                    # Display overall metrics
                    st.markdown("#### Overall Metrics")
                    
                    metrics_cols = st.columns(4)
                    metrics = eval_results['overall_metrics']
                    
                    for i, (metric, value) in enumerate(metrics.items()):
                        col_idx = i % 4
                        metrics_cols[col_idx].metric(metric.upper(), f"{value:.4f}")
                    
                    # Metric distributions
                    st.markdown("#### Metric Distributions")
                    
                    metric_tabs = st.tabs([
                        "NDCG@10", "MAP@10", "MRR", "Precision@10", "Recall@10"
                    ])
                    
                    with metric_tabs[0]:
                        fig = plot_metric_distribution(
                            eval_results['results_df'], 
                            'ndcg@10',
                            use_plotly=True
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"ndcg10_dist_{selected_detail_model}")
                    
                    with metric_tabs[1]:
                        fig = plot_metric_distribution(
                            eval_results['results_df'], 
                            'map@10',
                            use_plotly=True
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"map10_dist_{selected_detail_model}")
                    
                    with metric_tabs[2]:
                        fig = plot_metric_distribution(
                            eval_results['results_df'], 
                            'mrr',
                            use_plotly=True
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"mrr_dist_{selected_detail_model}")
                    
                    with metric_tabs[3]:
                        fig = plot_metric_distribution(
                            eval_results['results_df'], 
                            'precision@10',
                            use_plotly=True
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"precision10_dist_{selected_detail_model}")
                    
                    with metric_tabs[4]:
                        fig = plot_metric_distribution(
                            eval_results['results_df'], 
                            'recall@10',
                            use_plotly=True
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"recall10_dist_{selected_detail_model}")
                    
                    # Feature importance
                    st.markdown("#### Feature Importance")
                    
                    importance_df = eval_results['importance_df']
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        from utils.feature_analysis import plot_feature_importance
                        fig = plot_feature_importance(importance_df, top_n=15, use_plotly=True)
                        st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_{selected_detail_model}")
                    
                    with col2:
                        st.dataframe(
                            importance_df.head(15).reset_index(drop=True),
                            column_config={
                                "feature_index": st.column_config.NumberColumn("Feature", format="%d"),
                                "importance": st.column_config.NumberColumn("Importance", format="%.4f")
                            },
                            height=500
                        )
                    
                    # Per-query analysis
                    st.markdown("#### Query-Level Analysis")
                    
                    # Select a query to analyze
                    unique_qids = st.session_state['test_df']['query_id'].unique()
                    
                    selected_qid = st.selectbox(
                        "Select a query to analyze:",
                        options=unique_qids,
                        index=0
                    )
                    
                    if selected_qid:
                        # Get model predictions for this query
                        model = model_info['model']
                        model_name = model_info['name']
                        selected_features = model_info['features']
                        
                        predictions_df = get_predictions_by_query(
                            model,
                            st.session_state['test_df'],
                            model_name,
                            selected_qid,
                            features_to_use=selected_features
                        )
                        
                        if predictions_df is not None and not predictions_df.empty:
                            st.markdown(f"##### Results for Query {selected_qid}")
                            
                            # Display results ranked by model prediction
                            st.markdown("Ranked Results (Top 20):")
                            
                            # Add ranking position
                            predictions_df = predictions_df.reset_index(drop=True)
                            predictions_df.index = predictions_df.index + 1  # Start from rank 1
                            
                            # Format for display
                            display_df = predictions_df.head(20).copy()
                            display_df.columns = ['Document ID', 'True Relevance', 'Predicted Score']
                            
                            # Highlight rows based on relevance
                            def color_relevance(val):
                                if val == 4:
                                    return 'background-color: #d4f1dd'
                                elif val == 3:
                                    return 'background-color: #e6f7e9'
                                elif val == 0:
                                    return 'background-color: #ffe6e6'
                                return ''
                            
                            styled_df = display_df.style.applymap(
                                color_relevance, 
                                subset=['True Relevance']
                            )
                            
                            st.dataframe(styled_df)
                            
                            # Calculate metrics for this query
                            query_results = eval_results['results_df']
                            query_metrics = query_results[query_results['query_id'] == selected_qid].iloc[0]
                            
                            # Display metrics for this query
                            st.markdown("Metrics for this query:")
                            
                            metric_cols = st.columns(5)
                            metric_cols[0].metric("NDCG@10", f"{query_metrics['ndcg@10']:.4f}")
                            metric_cols[1].metric("MAP@10", f"{query_metrics['map@10']:.4f}")
                            metric_cols[2].metric("MRR", f"{query_metrics['mrr']:.4f}")
                            metric_cols[3].metric("Precision@10", f"{query_metrics['precision@10']:.4f}")
                            metric_cols[4].metric("Recall@10", f"{query_metrics['recall@10']:.4f}")
                            
                            # Relevance distribution in top-k
                            top_k = min(20, len(predictions_df))
                            rel_counts = predictions_df.head(top_k)['relevance'].value_counts().sort_index()
                            
                            st.markdown(f"##### Relevance Distribution in Top {top_k} Results")
                            
                            fig = px.pie(
                                values=rel_counts.values,
                                names=rel_counts.index,
                                title=f'Relevance Distribution in Top {top_k} Results',
                                color_discrete_sequence=px.colors.sequential.Viridis
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True, key=f"rel_dist_{selected_qid}")
                        else:
                            st.error(f"No predictions available for query {selected_qid}")
                    
                    # Training history
                    if selected_detail_model in st.session_state['training_history']:
                        st.markdown("#### Training History")
                        
                        history = st.session_state['training_history'][selected_detail_model]
                        model_name = model_info['name']
                        
                        fig = plot_training_history(history, model_name, use_plotly=True)
                        st.plotly_chart(fig, use_container_width=True, key=f"training_history_{selected_detail_model}")
            else:
                st.warning("No evaluation results available. Please evaluate models first.")
    else:
        st.info("Please select at least one model to evaluate.")
else:
    st.info("Please load the test data to begin model evaluation.")

# Model Comparison Section
st.header("Model Comparison")

if 'evaluation_results' in st.session_state and st.session_state['evaluation_results']:
    # Get models that have been evaluated for the current dataset
    evaluated_models = {
        model_id: dataset_models[model_id]['name']
        for model_id in st.session_state['evaluation_results'].keys()
        if model_id in dataset_models
    }
    
    if evaluated_models:
        st.subheader("Compare Models")
        
        # Select models to compare
        compare_model_ids = st.multiselect(
            "Select models to compare:",
            options=list(evaluated_models.keys()),
            default=list(evaluated_models.keys())[:min(2, len(evaluated_models))],
            format_func=lambda x: f"{evaluated_models[x]} ({x.split('_')[-1]})"
        )
        
        if len(compare_model_ids) >= 2:
            # Extract metrics for selected models
            metrics_list = [
                st.session_state['evaluation_results'][model_id]['overall_metrics']
                for model_id in compare_model_ids
            ]
            
            model_names = [evaluated_models[model_id] for model_id in compare_model_ids]
            
            # Plot metrics comparison
            st.markdown("### Metrics Comparison")
            
            fig = plot_metrics_comparison(metrics_list, model_names, use_plotly=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance comparison
            st.markdown("### Feature Importance Comparison")
            
            importance_dfs = [
                st.session_state['evaluation_results'][model_id]['importance_df']
                for model_id in compare_model_ids
            ]
            
            fig = plot_feature_importance_comparison(
                importance_dfs, 
                model_names, 
                top_n=10,
                use_plotly=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Per-metric comparison
            st.markdown("### Performance by Metric")
            
            metric_tabs = st.tabs([
                "NDCG@k", "Precision & Recall", "MRR & MAP"
            ])
            
            with metric_tabs[0]:
                # NDCG at different k values
                ndcg_data = []
                
                for i, model_id in enumerate(compare_model_ids):
                    metrics = st.session_state['evaluation_results'][model_id]['overall_metrics']
                    model_name = evaluated_models[model_id]
                    
                    ndcg_data.extend([
                        {'Model': model_name, 'Metric': 'NDCG@1', 'Value': metrics['ndcg@1']},
                        {'Model': model_name, 'Metric': 'NDCG@3', 'Value': metrics['ndcg@3']},
                        {'Model': model_name, 'Metric': 'NDCG@5', 'Value': metrics['ndcg@5']},
                        {'Model': model_name, 'Metric': 'NDCG@10', 'Value': metrics['ndcg@10']}
                    ])
                
                ndcg_df = pd.DataFrame(ndcg_data)
                
                fig = px.bar(
                    ndcg_df,
                    x='Metric',
                    y='Value',
                    color='Model',
                    barmode='group',
                    title='NDCG at Different Cutoffs',
                    labels={'Value': 'Score'},
                    height=500
                )
                
                fig.update_layout(
                    xaxis={'categoryorder': 'category ascending'},
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with metric_tabs[1]:
                # Precision and Recall
                pr_data = []
                
                for i, model_id in enumerate(compare_model_ids):
                    metrics = st.session_state['evaluation_results'][model_id]['overall_metrics']
                    model_name = evaluated_models[model_id]
                    
                    pr_data.extend([
                        {'Model': model_name, 'Metric': 'Precision@10', 'Value': metrics['precision@10']},
                        {'Model': model_name, 'Metric': 'Recall@10', 'Value': metrics['recall@10']}
                    ])
                
                pr_df = pd.DataFrame(pr_data)
                
                fig = px.bar(
                    pr_df,
                    x='Metric',
                    y='Value',
                    color='Model',
                    barmode='group',
                    title='Precision and Recall',
                    labels={'Value': 'Score'},
                    height=500
                )
                
                fig.update_layout(
                    xaxis={'categoryorder': 'category ascending'},
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with metric_tabs[2]:
                # MRR and MAP
                mrr_map_data = []
                
                for i, model_id in enumerate(compare_model_ids):
                    metrics = st.session_state['evaluation_results'][model_id]['overall_metrics']
                    model_name = evaluated_models[model_id]
                    
                    mrr_map_data.extend([
                        {'Model': model_name, 'Metric': 'MRR', 'Value': metrics['mrr']},
                        {'Model': model_name, 'Metric': 'MAP@10', 'Value': metrics['map@10']}
                    ])
                
                mrr_map_df = pd.DataFrame(mrr_map_data)
                
                fig = px.bar(
                    mrr_map_df,
                    x='Metric',
                    y='Value',
                    color='Model',
                    barmode='group',
                    title='MRR and MAP',
                    labels={'Value': 'Score'},
                    height=500
                )
                
                fig.update_layout(
                    xaxis={'categoryorder': 'category ascending'},
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Per-query performance comparison
            st.markdown("### Query-Level Performance Comparison")
            
            # Select a specific query to compare performance
            if 'test_df' in st.session_state:
                unique_qids = st.session_state['test_df']['query_id'].unique()
                
                selected_qid = st.selectbox(
                    "Select a query to compare:",
                    options=unique_qids,
                    index=0,
                    key="compare_query_selector"
                )
                
                if selected_qid:
                    st.markdown(f"#### Performance Comparison for Query {selected_qid}")
                    
                    # Extract metrics for this query across models
                    query_metrics = []
                    
                    for model_id in compare_model_ids:
                        model_name = evaluated_models[model_id]
                        results_df = st.session_state['evaluation_results'][model_id]['results_df']
                        
                        query_row = results_df[results_df['query_id'] == selected_qid]
                        
                        if not query_row.empty:
                            metrics = query_row.iloc[0].to_dict()
                            metrics['Model'] = model_name
                            query_metrics.append(metrics)
                    
                    if query_metrics:
                        # Create radar chart for comparing models on this query
                        metrics_to_plot = ['ndcg@10', 'map@10', 'mrr', 'precision@10', 'recall@10']
                        
                        fig = go.Figure()
                        
                        for metrics in query_metrics:
                            model_name = metrics['Model']
                            
                            fig.add_trace(go.Scatterpolar(
                                r=[metrics[m] for m in metrics_to_plot],
                                theta=[m.upper() for m in metrics_to_plot],
                                fill='toself',
                                name=model_name
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title=f"Performance Metrics for Query {selected_qid}",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Also display rankings side by side
                        st.markdown("#### Top 10 Results Comparison")
                        
                        # Get predictions for each model
                        all_predictions = []
                        
                        for model_id in compare_model_ids:
                            model_info = dataset_models[model_id]
                            model = model_info['model']
                            model_name = model_info['name']
                            selected_features = model_info['features']
                            
                            predictions_df = get_predictions_by_query(
                                model,
                                st.session_state['test_df'],
                                model_name,
                                selected_qid,
                                features_to_use=selected_features
                            )
                            
                            if predictions_df is not None and not predictions_df.empty:
                                # Keep top 10 results
                                top_results = predictions_df.head(10).copy()
                                top_results['Model'] = model_name
                                all_predictions.append(top_results)
                        
                        if all_predictions:
                            # Display side by side
                            cols = st.columns(len(all_predictions))
                            
                            for i, (col, pred_df) in enumerate(zip(cols, all_predictions)):
                                model_name = compare_model_ids[i]
                                display_name = evaluated_models[model_name]
                                
                                col.markdown(f"**{display_name}**")
                                
                                # Format for display
                                display_df = pred_df.reset_index(drop=True).copy()
                                display_df.index = display_df.index + 1  # Start from rank 1
                                display_df = display_df[['document_id', 'relevance']].copy()
                                display_df.columns = ['Doc ID', 'Relevance']
                                
                                # Highlight rows based on relevance
                                def color_relevance(val):
                                    if val == 4:
                                        return 'background-color: #d4f1dd'
                                    elif val == 3:
                                        return 'background-color: #e6f7e9'
                                    elif val == 0:
                                        return 'background-color: #ffe6e6'
                                    return ''
                                
                                styled_df = display_df.style.applymap(
                                    color_relevance, 
                                    subset=['Relevance']
                                )
                                
                                col.dataframe(styled_df, height=400)
                    else:
                        st.warning(f"No metrics available for query {selected_qid}")
        else:
            st.info("Please select at least two models to compare.")
    else:
        st.warning("No evaluated models available for the current dataset.")
else:
    st.info("No evaluation results available. Please evaluate models first.")

# Error Analysis Section
st.header("Error Analysis")

if 'evaluation_results' in st.session_state and st.session_state['evaluation_results']:
    # Get models that have been evaluated for the current dataset
    evaluated_models = {
        model_id: dataset_models[model_id]['name']
        for model_id in st.session_state['evaluation_results'].keys()
        if model_id in dataset_models
    }
    
    if evaluated_models:
        # Select a model for error analysis
        selected_error_model = st.selectbox(
            "Select a model for error analysis:",
            options=list(evaluated_models.keys()),
            format_func=lambda x: f"{evaluated_models[x]} ({x.split('_')[-1]})",
            key="error_analysis_model_selector"
        )
        
        if selected_error_model:
            model_info = dataset_models[selected_error_model]
            model_name = model_info['name']
            eval_results = st.session_state['evaluation_results'][selected_error_model]
            
            st.subheader(f"Error Analysis for {model_name}")
            
            # Find queries with poor performance
            results_df = eval_results['results_df']
            
            # Sort by NDCG@10 ascending to find worst performing queries
            worst_queries = results_df.sort_values('ndcg@10').head(10)
            
            st.markdown("#### Worst Performing Queries")
            
            # Display worst queries
            st.dataframe(
                worst_queries[['query_id', 'ndcg@10', 'map@10', 'mrr', 'document_count']],
                column_config={
                    "query_id": st.column_config.NumberColumn("Query ID"),
                    "ndcg@10": st.column_config.NumberColumn("NDCG@10", format="%.4f"),
                    "map@10": st.column_config.NumberColumn("MAP@10", format="%.4f"),
                    "mrr": st.column_config.NumberColumn("MRR", format="%.4f"),
                    "document_count": st.column_config.NumberColumn("Doc Count")
                }
            )
            
            # Select a query for detailed error analysis
            if not worst_queries.empty:
                selected_error_qid = st.selectbox(
                    "Select a query for detailed error analysis:",
                    options=worst_queries['query_id'].tolist(),
                    index=0,
                    key="error_query_selector"
                )
                
                if selected_error_qid and 'test_df' in st.session_state:
                    st.markdown(f"#### Error Analysis for Query {selected_error_qid}")
                    
                    # Get model predictions for this query
                    model = model_info['model']
                    selected_features = model_info['features']
                    
                    predictions_df = get_predictions_by_query(
                        model,
                        st.session_state['test_df'],
                        model_name,
                        selected_error_qid,
                        features_to_use=selected_features
                    )
                    
                    if predictions_df is not None and not predictions_df.empty:
                        # Display side by side comparison of model ranking vs. true ranking
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Model Ranking")
                            
                            # Format for display
                            model_ranking = predictions_df.reset_index(drop=True).copy()
                            model_ranking.index = model_ranking.index + 1  # Start from rank 1
                            model_ranking = model_ranking[['document_id', 'relevance', 'prediction']].head(20)
                            model_ranking.columns = ['Doc ID', 'True Relevance', 'Predicted Score']
                            
                            # Highlight rows based on relevance
                            def color_relevance(val):
                                if val == 4:
                                    return 'background-color: #d4f1dd'
                                elif val == 3:
                                    return 'background-color: #e6f7e9'
                                elif val == 0:
                                    return 'background-color: #ffe6e6'
                                return ''
                            
                            styled_df = model_ranking.style.applymap(
                                color_relevance, 
                                subset=['True Relevance']
                            )
                            
                            st.dataframe(styled_df, height=500)
                        
                        with col2:
                            st.markdown("##### Ideal Ranking (by true relevance)")
                            
                            # Sort by true relevance
                            ideal_ranking = predictions_df.sort_values('relevance', ascending=False).reset_index(drop=True).copy()
                            ideal_ranking.index = ideal_ranking.index + 1  # Start from rank 1
                            ideal_ranking = ideal_ranking[['document_id', 'relevance', 'prediction']].head(20)
                            ideal_ranking.columns = ['Doc ID', 'True Relevance', 'Predicted Score']
                            
                            # Highlight rows based on relevance
                            styled_df = ideal_ranking.style.applymap(
                                color_relevance, 
                                subset=['True Relevance']
                            )
                            
                            st.dataframe(styled_df, height=500)
                        
                        # Analysis of misplaced documents
                        st.markdown("##### Misplaced Documents Analysis")
                        
                        # Find the most relevant documents that are ranked too low (errors of omission)
                        high_rel_docs = predictions_df[predictions_df['relevance'] >= 3].copy()
                        misplaced_docs = high_rel_docs[high_rel_docs.index >= 10]  # Relevant docs not in top 10
                        
                        if not misplaced_docs.empty:
                            st.warning(f"Found {len(misplaced_docs)} highly relevant documents (relevance ≥ 3) ranked outside top 10")
                            
                            # Display these documents
                            misplaced_df = misplaced_docs.reset_index(drop=True).copy()
                            misplaced_df['Rank'] = misplaced_docs.index + 1
                            misplaced_df = misplaced_df[['Rank', 'document_id', 'relevance', 'prediction']]
                            misplaced_df.columns = ['Model Rank', 'Doc ID', 'True Relevance', 'Predicted Score']
                            
                            st.dataframe(misplaced_df, height=200)
                        else:
                            st.success("All highly relevant documents (relevance ≥ 3) are in the top 10 results")
                        
                        # Find irrelevant documents ranked too high (errors of commission)
                        top_docs = predictions_df.head(10).copy()
                        irrelevant_top = top_docs[top_docs['relevance'] == 0]
                        
                        if not irrelevant_top.empty:
                            st.warning(f"Found {len(irrelevant_top)} irrelevant documents (relevance = 0) in the top 10")
                            
                            # Display these documents
                            irrelevant_df = irrelevant_top.reset_index(drop=True).copy()
                            irrelevant_df['Rank'] = irrelevant_top.index + 1
                            irrelevant_df = irrelevant_df[['Rank', 'document_id', 'relevance', 'prediction']]
                            irrelevant_df.columns = ['Model Rank', 'Doc ID', 'True Relevance', 'Predicted Score']
                            
                            st.dataframe(irrelevant_df, height=200)
                        else:
                            st.success("No irrelevant documents (relevance = 0) in the top 10 results")
                    else:
                        st.error(f"No predictions available for query {selected_error_qid}")
            else:
                st.warning("No results available for error analysis")
    else:
        st.warning("No evaluated models available for the current dataset.")
else:
    st.info("No evaluation results available. Please evaluate models first.")

# Update feature importance visualization
st.subheader("Feature Importance Analysis")
st.markdown("""
Feature importance is analyzed across the following groups:

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
