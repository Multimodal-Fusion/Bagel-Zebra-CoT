#!/usr/bin/env python3
"""
Convert Zebra-CoT Chess dataset to ThinkTrace JSONL format.
Generates 1000 samples for training.
"""

import json
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import re
import argparse
from tqdm import tqdm


def extract_and_save_images(row, sample_idx, output_images_dir):
    """Extract images from row and save them, returning image references."""
    image_refs = {}
    
    # Extract problem image
    if 'problem_image_1' in row and row['problem_image_1'] is not None:
        img_bytes = row['problem_image_1']['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        img_path = output_images_dir / f"chess_{sample_idx:04d}_problem_image_1.png"
        img.save(img_path)
        image_refs['problem_image_1'] = str(img_path.name)
    
    # Extract reasoning images (up to 10)
    for i in range(1, 11):
        col_name = f'reasoning_image_{i}'
        if col_name in row and row[col_name] is not None:
            img_bytes = row[col_name]['bytes']
            img = Image.open(io.BytesIO(img_bytes))
            img_path = output_images_dir / f"chess_{sample_idx:04d}_reasoning_image_{i}.png"
            img.save(img_path)
            image_refs[col_name] = str(img_path.name)
    
    return image_refs


def convert_sample(row, sample_idx, output_images_dir):
    """Convert a single Zebra-CoT sample to ThinkTrace format."""
    
    # Extract and save images
    image_refs = extract_and_save_images(row, sample_idx, output_images_dir)
    
    # Build the ThinkTrace format
    thinktrace_sample = {
        "Question": row.get('Question', ''),
        "Text Reasoning Trace": row.get('Text Reasoning Trace', ''),
        "Final Answer": row.get('Final Answer', ''),
    }
    
    # Add image references
    thinktrace_sample.update(image_refs)
    
    # Process text to add image references
    question = thinktrace_sample["Question"]
    if 'problem_image_1' in image_refs:
        # Add image reference to question if not already present
        if '<image_start>' not in question:
            question = f"{question}\n<image_start>[problem_image_1]<image_end>"
            thinktrace_sample["Question"] = question
    
    # Process reasoning trace to add image references
    reasoning = thinktrace_sample["Text Reasoning Trace"]
    for i in range(1, 11):
        img_key = f'reasoning_image_{i}'
        if img_key in image_refs:
            # Add image reference inline with reasoning
            pattern = f"THOUGHT {i}:"
            if pattern in reasoning and f'<image_start>[{img_key}]<image_end>' not in reasoning:
                reasoning = reasoning.replace(
                    pattern, 
                    f"{pattern}\n<image_start>[{img_key}]<image_end>\n"
                )
    thinktrace_sample["Text Reasoning Trace"] = reasoning
    
    return thinktrace_sample


def main():
    parser = argparse.ArgumentParser(description='Convert Zebra-CoT Chess to ThinkTrace format')
    parser.add_argument('--num_samples', type=int, default=1000, 
                       help='Number of samples to convert (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='datasets',
                       help='Output directory (default: datasets)')
    args = parser.parse_args()
    
    # Input and output paths
    parquet_dir = Path("datasets/Zebra-CoT/Visual Logic & Strategic Games - Chess")
    output_jsonl = Path(args.output_dir) / "chess_thinktrace.jsonl"
    output_images_dir = Path(args.output_dir) / "chess_thinktrace_images"
    
    # Create output directory for images
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all parquet files
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load samples from multiple files if needed
    all_samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_samples.extend(df.to_dict('records'))
        print(f"Loaded {len(df)} samples from {pf.name}, total: {len(all_samples)}")
        if len(all_samples) >= args.num_samples:
            break
    
    # Limit to requested number of samples
    samples_to_convert = min(args.num_samples, len(all_samples))
    print(f"\nConverting {samples_to_convert} samples to ThinkTrace format...")
    
    # Convert samples
    with open(output_jsonl, 'w') as f:
        for idx in tqdm(range(samples_to_convert), desc="Converting samples"):
            sample = all_samples[idx]
            
            try:
                thinktrace = convert_sample(sample, idx, output_images_dir)
                
                # Write JSONL
                json_line = json.dumps(thinktrace, ensure_ascii=False)
                f.write(json_line + '\n')
                
                if idx < 3:  # Show first few samples
                    print(f"\nSample {idx}:")
                    print(f"  Question: {thinktrace['Question'][:100]}...")
                    print(f"  Images saved: {len([k for k in thinktrace.keys() if 'image' in k])}")
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    print(f"\n✅ Conversion complete!")
    print(f"  Output JSONL: {output_jsonl}")
    print(f"  Images directory: {output_images_dir}")
    print(f"  Total samples: {samples_to_convert}")
    
    # Count images
    image_count = len(list(output_images_dir.glob("*.png")))
    print(f"  Total images: {image_count}")


if __name__ == "__main__":
    main()