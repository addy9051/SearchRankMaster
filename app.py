import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import check_dataset_availability, load_dataset, load_dataset_info

# Set page configuration
st.set_page_config(
    page_title="ML-Powered Search Ranking System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page header
st.title("Machine Learning-Powered Search Ranking System")
st.markdown("""
This application allows you to develop, train, and evaluate machine learning models 
for search ranking using the Microsoft Learning to Rank dataset (MSLR-WEB10K).
""")

# Environment info
st.sidebar.header("Environment Info")
st.sidebar.text("Using local MSLR-WEB10K dataset")
st.sidebar.text("Using scikit-learn for ML models")
st.sidebar.text("Using pandas for data handling")

# Dataset info
st.sidebar.header("Dataset Info")
datasets_available = check_dataset_availability()

if not datasets_available:
    st.sidebar.error("❌ MSLR dataset files are not found in the local directory.")
    st.sidebar.info("""
    To use this application, you need to ensure the following files are present in the root directory:
    - small_train.csv (Training data)
    - small_valid.csv (Validation data)
    - small_test.csv (Test data)
    
    These files contain the MSLR-WEB10K dataset split into training, validation, and test sets.
    """)
else:
    st.sidebar.success("✅ MSLR dataset files are available in the local directory.")
    
    # Load and display dataset info
    try:
        dataset_info = load_dataset_info('mslr_web10k')
        if dataset_info:
            st.sidebar.markdown("### Dataset Details")
            st.sidebar.markdown(f"**Name**: {dataset_info['name']}")
            st.sidebar.markdown(f"**Description**: {dataset_info['description']}")
            st.sidebar.markdown(f"**Total Queries**: {dataset_info['num_queries']}")
            st.sidebar.markdown(f"**Total Features**: {dataset_info['features']['total_features']}")
    except Exception as e:
        st.sidebar.error(f"Error loading dataset info: {str(e)}")
        
        # Display feature groups if available
        if 'features' in dataset_info and 'feature_groups' in dataset_info['features']:
            st.sidebar.markdown("### Feature Groups")
            for group, description in dataset_info['features']['feature_groups'].items():
                st.sidebar.markdown(f"**{group.title()}**: {description}")
        else:
            st.sidebar.warning("Feature groups information not available")

# Main navigation for the home page
st.header("Getting Started")
st.markdown("""
### Project Roadmap
1. **Data Exploration**: Analyze and understand the MSLR dataset, its features, and distributions.
2. **Feature Analysis**: Investigate feature importance, correlations, and impact on rankings.
3. **Model Training**: Train various Learning to Rank models using different approaches.
4. **Model Evaluation**: Evaluate models using standard ranking metrics and compare performance.

### About the Dataset
The Microsoft Learning to Rank dataset (MSLR-WEB10K) is a benchmark dataset for research 
on learning to rank algorithms. It contains:
- Query-document pairs with relevance judgments (0-4)
- 136 features per query-document pair, organized into groups:
  - Document fields (body, anchor, title, URL, whole document)
  - URL structure features
  - Link features (inlinks and outlinks)
  - Ranking features (PageRank and SiteRank)
  - Quality scores
  - Click-based features

Each row in the dataset represents a query-document pair with:
- A relevance label (0-4)
- A query ID
- 136 features extracted from the query-document pair
""")

# Dataset selection section
st.header("Select Dataset")
st.info("### MSLR-WEB10K")
st.markdown("- 10,000 queries\n- 136 features per query-document pair\n- 5-level relevance judgments")
if st.button("Explore MSLR-WEB10K", key="explore_10k"):
    if datasets_available:
        st.session_state['selected_dataset'] = 'mslr_web10k'
        st.rerun()
    else:
        st.error("Please ensure the MSLR dataset files are present in the root directory.")

# Learning to Rank approaches information
st.header("Learning to Rank Approaches")
approach_col1, approach_col2, approach_col3 = st.columns(3)

with approach_col1:
    st.subheader("Pointwise Approach")
    st.markdown("""
    - Treats each query-document pair independently
    - Predicts absolute relevance scores
    - Uses regression or classification algorithms
    - Examples: Linear Regression, Random Forest
    """)

with approach_col2:
    st.subheader("Pairwise Approach")
    st.markdown("""
    - Compares pairs of documents for a given query
    - Learns relative preferences between documents
    - Examples: RankNet, LambdaMART, SVMRank
    """)
    
with approach_col3:
    st.subheader("Listwise Approach")
    st.markdown("""
    - Directly optimizes ranking metrics over entire lists
    - Considers the complete ranking context
    - Examples: ListNet, AdaRank, TF-Ranking
    """)

st.markdown("---")
st.markdown("""
### References:
- [Microsoft Learning to Rank Datasets](https://www.microsoft.com/en-us/research/project/mslr/)
- [Learning to Rank Overview](https://en.wikipedia.org/wiki/Learning_to_rank)
""")
