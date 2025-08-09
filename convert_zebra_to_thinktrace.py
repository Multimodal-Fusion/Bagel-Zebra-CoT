#!/usr/bin/env python3
"""
Convert Zebra-CoT parquet files to ThinkTrace JSONL format.
This script handles the Chess dataset as an example.
"""

import pandas as pd
import json
import base64
from pathlib import Path
from PIL import Image
import io
from typing import Dict, List, Any
import re

def save_image_from_bytes(image_bytes: bytes, output_path: Path) -> str:
    """Save image bytes to file and return the filename."""
    img = Image.open(io.BytesIO(image_bytes))
    img.save(output_path)
    return output_path.name

def extract_image_references_from_trace(trace: str) -> List[str]:
    """Extract which reasoning images are referenced in the trace."""
    # Look for patterns like "Let me visualize" or "I'll draw" or image references
    image_refs = []
    
    # Split by THOUGHT markers
    thoughts = re.split(r'THOUGHT \d+:', trace)
    
    for i, thought in enumerate(thoughts[1:], 1):  # Skip first empty split
        # Simple heuristic: if thought mentions visualization/drawing/image
        if any(keyword in thought.lower() for keyword in 
               ['visualiz', 'draw', 'diagram', 'illustrat', 'image', 'picture', 'show']):
            image_refs.append(f'reasoning_image_{i}')
    
    return image_refs

def convert_zebra_to_thinktrace(
    parquet_file: Path,
    output_jsonl: Path,
    images_dir: Path,
    num_samples: int = 5
) -> None:
    """
    Convert Zebra-CoT parquet to ThinkTrace JSONL format.
    
    Args:
        parquet_file: Path to input parquet file
        output_jsonl: Path to output JSONL file
        images_dir: Directory to save extracted images
        num_samples: Number of samples to convert
    """
    
    # Create images directory if it doesn't exist
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parquet file
    df = pd.read_parquet(parquet_file)
    df = df.head(num_samples)  # Limit samples for demo
    
    print(f"Converting {len(df)} samples from {parquet_file.name}")
    
    converted_samples = []
    
    for idx, row in df.iterrows():
        sample_id = f"chess_{idx:04d}"
        
        # Extract fields
        question = row['Question']
        reasoning_trace = row['Text Reasoning Trace']
        final_answer = row['Final Answer']
        
        # Process images
        saved_images = {}
        
        # Save problem images
        for img_col in ['problem_image_1', 'problem_image_2']:
            if img_col in row and row[img_col] is not None:
                img_bytes = row[img_col]['bytes'] if isinstance(row[img_col], dict) else row[img_col]
                if img_bytes:
                    img_path = images_dir / f"{sample_id}_{img_col}.png"
                    img_name = save_image_from_bytes(img_bytes, img_path)
                    saved_images[img_col] = img_name
        
        # Save reasoning images
        for i in range(1, 5):  # Up to 4 reasoning images
            img_col = f'reasoning_image_{i}'
            if img_col in row and row[img_col] is not None:
                img_bytes = row[img_col]['bytes'] if isinstance(row[img_col], dict) else row[img_col]
                if img_bytes:
                    img_path = images_dir / f"{sample_id}_{img_col}.png"
                    img_name = save_image_from_bytes(img_bytes, img_path)
                    saved_images[img_col] = img_name
        
        # Format question with image references
        formatted_question = question
        if 'problem_image_1' in saved_images:
            formatted_question = f"<image_start>[problem_image_1]<image_end> {formatted_question}"
        if 'problem_image_2' in saved_images:
            formatted_question += f" <image_start>[problem_image_2]<image_end>"
        
        # Format reasoning trace with image references
        # Detect where images should be inserted based on thought patterns
        formatted_trace = reasoning_trace
        
        # Insert reasoning images at appropriate points
        thought_parts = re.split(r'(THOUGHT \d+:)', formatted_trace)
        reconstructed_trace = ""
        thought_counter = 0
        
        for part in thought_parts:
            if re.match(r'THOUGHT \d+:', part):
                thought_counter += 1
                reconstructed_trace += part
                # Check if we have a reasoning image for this thought
                img_key = f'reasoning_image_{thought_counter}'
                if img_key in saved_images:
                    # Add image reference after thought header
                    reconstructed_trace += f" <image_start>[{img_key}]<image_end>"
            else:
                reconstructed_trace += part
        
        # Create ThinkTrace format
        thinktrace_sample = {
            "Question": formatted_question,
            "Text Reasoning Trace": reconstructed_trace,
            "Final Answer": final_answer,
            **saved_images  # Add image paths
        }
        
        converted_samples.append(thinktrace_sample)
        print(f"  Converted sample {idx}: {len(saved_images)} images saved")
    
    # Write to JSONL
    with open(output_jsonl, 'w') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\nConversion complete! Wrote {len(converted_samples)} samples to {output_jsonl}")
    print(f"Images saved to {images_dir}")
    
    return converted_samples

def main():
    # Paths
    chess_parquet = Path("./datasets/Zebra-CoT/Visual Logic & Strategic Games - Chess/train-00000-of-00008.parquet")
    output_jsonl = Path("./datasets/chess_thinktrace.jsonl")
    images_dir = Path("./datasets/chess_thinktrace_images")
    
    # Convert
    samples = convert_zebra_to_thinktrace(
        chess_parquet,
        output_jsonl,
        images_dir,
        num_samples=3  # Just 3 samples for demo
    )
    
    # Display first sample
    if samples:
        print("\n" + "="*80)
        print("FIRST CONVERTED SAMPLE:")
        print("="*80)
        first = samples[0]
        print(f"Question: {first['Question'][:200]}...")
        print(f"Reasoning (first 300 chars): {first['Text Reasoning Trace'][:300]}...")
        print(f"Final Answer: {first['Final Answer'][:200]}...")
        print(f"Images: {[k for k in first.keys() if 'image' in k]}")

if __name__ == "__main__":
    main()