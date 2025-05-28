# SearchRankMaster

SearchRankMaster is a Python library for learning-to-rank (LTR) tasks, designed to help you build and evaluate ranking models for search and recommendation systems.

## Features

- **Flexible Data Loading**: Supports various data formats (CSV, Parquet) and handles large datasets with chunking and Dask integration.
- **Multiple Ranking Models**: Includes implementations of popular ranking algorithms like LightGBM, XGBoost, and linear models.
- **Comprehensive Evaluation**: Provides a wide range of ranking metrics including NDCG, MAP, MRR, Kendall's tau, and Spearman's rho.
- **Easy-to-Use API**: Simple and intuitive interface for training, evaluating, and deploying ranking models.
- **Visualization**: Built-in support for visualizing model performance and feature importance.

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/search-rank-master.git
cd search-rank-master
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/yourusername/search-rank-master.git
```

## Quick Start

```python
import numpy as np
import pandas as pd
from searchrankmaster import DataLoader, create_model, evaluate_ranking

# Load data
loader = DataLoader('mslr_web10k')
train_data = loader.load_split('train')
X_train, y_train, qids_train = loader.prepare_dataset_for_training(train_data)

# Create and train a model
model = create_model('lightgbm', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train, qids_train)

# Make predictions
X_test, y_test, qids_test = loader.prepare_dataset_for_training(loader.load_split('test'))
y_pred = model.predict(X_test)

# Evaluate
results = evaluate_ranking(y_test, y_pred, qids_test)
print(results)
```

## Documentation

### Data Loading

```python
from searchrankmaster import DataLoader

# Initialize with a dataset name and optional data directory
loader = DataLoader('mslr_web10k', data_dir='./data')

# Load a specific split
train_data = loader.load_split('train')
valid_data = loader.load_split('validation')
test_data = loader.load_split('test')

# Load in chunks (for large datasets)
for chunk in loader.load_in_chunks('train', chunk_size=1000):
    # Process each chunk
    pass
```

### Model Training

```python
from searchrankmaster import create_model

# Create a LightGBM ranker
model = create_model('lightgbm', 
                   objective='lambdarank',
                   metric='ndcg',
                   n_estimators=100,
                   learning_rate=0.1)

# Train the model
model.fit(X_train, y_train, qids_train)

# Save the model
model.save('model.joblib')

# Load a saved model
from searchrankmaster import BaseRanker
model = BaseRanker.load('model.joblib')
```

### Evaluation

```python
from searchrankmaster import evaluate_ranking, plot_metrics

# Evaluate with default metrics
results = evaluate_ranking(y_true, y_pred, qids)
print(results)

# Custom metrics
metrics = {
    'ndcg@5': {'func': 'ndcg_score', 'k': 5},
    'map@10': {'func': 'map_score', 'k': 10},
    'mrr': {'func': 'mrr_score'}
}
results = evaluate_ranking(y_true, y_pred, qids, metrics=metrics)

# Plot metrics
fig = plot_metrics(y_true, y_pred, qids)
fig.show()
```

## Available Models

- **LightGBM**: Gradient boosting framework with ranking objectives
- **XGBoost**: Scalable and flexible gradient boosting
- **Linear**: Simple linear model with L1/L2 regularization

## Evaluation Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain at k
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **Kendall's τ**: Rank correlation coefficient
- **Spearman's ρ**: Rank correlation coefficient

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
