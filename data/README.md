# SearchRankMaster - Data Directory

This directory contains the datasets used by SearchRankMaster for learning-to-rank tasks.

## Dataset Structure

```
data/
├── mslr_web10k/
│   ├── train.csv          # Training dataset (~70MB)
│   ├── test.csv           # Test dataset (~20MB) 
│   ├── validation.csv     # Validation dataset (~20MB)
│   └── README.md         # This file
└── sample/
    ├── sample_train.csv   # Small sample for testing (included in repo)
    └── sample_test.csv    # Small sample for testing (included in repo)
```

## Large Dataset Storage

Due to GitHub's file size limitations, the full datasets are stored externally:

### Option 1: Download from GitHub Releases
```bash
# Run the download script
python download_data.py
```

### Option 2: Manual Download
1. Download the full datasets from: [Release Page](https://github.com/yourusername/SearchRankMaster/releases)
2. Extract to the appropriate directories as shown above

### Option 3: Use Your Own Data
Replace the CSV files with your own learning-to-rank datasets following the same format:
- `qid`: Query ID
- `target`: Relevance score  
- `features`: Feature columns (numbered 1, 2, 3, etc.)

## Data Format

The expected CSV format:
```
qid,target,1,2,3,4,5,...
1,2,0.5,0.3,0.8,0.1,0.9,...
1,1,0.2,0.7,0.4,0.6,0.3,...
2,0,0.1,0.1,0.2,0.8,0.5,...
```

## Sample Data

Small sample datasets are included in the repository for testing and development purposes. These are located in the `sample/` directory.
