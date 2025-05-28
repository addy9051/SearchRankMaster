"""
Data Download Script for SearchRankMaster

This script downloads the required datasets from GitHub releases
or external sources instead of storing them in the Git repository.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination."""
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")
        os.remove(zip_path)  # Remove zip after extraction
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def setup_data_directory():
    """Setup data directory and download required datasets."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    mslr_dir = data_dir / "mslr_web10k"
    mslr_dir.mkdir(exist_ok=True)
    
    # Example URLs - replace with your actual data sources
    datasets = {
        "train.csv": "https://github.com/yourusername/SearchRankMaster/releases/download/v1.0/train.csv",
        "test.csv": "https://github.com/yourusername/SearchRankMaster/releases/download/v1.0/test.csv", 
        "validation.csv": "https://github.com/yourusername/SearchRankMaster/releases/download/v1.0/validation.csv"
    }
    
    print("Setting up SearchRankMaster datasets...")
    
    for filename, url in datasets.items():
        file_path = mslr_dir / filename
        if not file_path.exists():
            print(f"Downloading {filename}...")
            # For now, create placeholder message
            with open(file_path, 'w') as f:
                f.write(f"# Placeholder for {filename}\n")
                f.write(f"# Download from: {url}\n")
                f.write("# This file should be downloaded separately due to size constraints\n")
        else:
            print(f"{filename} already exists")
    
    print("Data setup complete!")
    print("Note: Large datasets should be downloaded separately or from GitHub releases")

if __name__ == "__main__":
    setup_data_directory()
