#!/usr/bin/env python3
"""Download Zebra-CoT dataset from Hugging Face"""

import os
import json
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

def download_zebra_cot():
    """Download the Zebra-CoT dataset"""
    
    # Create datasets directory
    dataset_dir = "./datasets/Zebra-CoT"
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("Downloading Zebra-CoT dataset from Hugging Face...")
    
    try:
        # Download the entire dataset
        cache_dir = snapshot_download(
            repo_id="multimodal-reasoning-lab/Zebra-CoT",
            repo_type="dataset",
            local_dir=dataset_dir,
            local_dir_use_symlinks=False
        )
        print(f"Dataset downloaded to: {dataset_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(dataset_dir):
            level = root.replace(dataset_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        return dataset_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_zebra_cot()