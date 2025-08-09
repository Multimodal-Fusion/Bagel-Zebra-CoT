#!/usr/bin/env python3
"""Explore Zebra-CoT dataset structure"""

import pandas as pd
import json
from pathlib import Path

def explore_dataset():
    dataset_dir = Path("./datasets/Zebra-CoT")
    
    # Find all parquet files
    parquet_files = list(dataset_dir.glob("**/*.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files")
    print("\nDataset categories:")
    
    categories = {}
    for pf in parquet_files:
        category = pf.parent.name
        if category not in categories:
            categories[category] = []
        categories[category].append(pf)
    
    for category, files in sorted(categories.items()):
        print(f"  {category}: {len(files)} files")
    
    # Sample one category to understand structure
    sample_category = "Scientific Reasoning - Physics"
    sample_file = next(dataset_dir.glob(f"{sample_category}/*.parquet"))
    
    print(f"\n\nExamining sample from '{sample_category}':")
    print(f"File: {sample_file.name}")
    
    df = pd.read_parquet(sample_file)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print(f"\nFirst sample:")
    if len(df) > 0:
        sample = df.iloc[0]
        for col, val in sample.items():
            if isinstance(val, str) and len(val) > 200:
                print(f"  {col}: {val[:200]}...")
            else:
                print(f"  {col}: {val}")
    
    return df

if __name__ == "__main__":
    df = explore_dataset()