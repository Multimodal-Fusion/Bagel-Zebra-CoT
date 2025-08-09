#!/usr/bin/env python3
"""
Visualize ThinkTrace dataset and dataloader for understanding the interleaved format.
Based on visualize_dataloader.py but specifically for ThinkTrace dataset.
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path
import json

# Add project roots to path
bagel_root = Path("/home/colligo/project/vlm/Bagel")
project_root = Path(__file__).parent
sys.path.insert(0, str(bagel_root))
sys.path.insert(0, str(project_root))

# Import from Bagel
sys.path.insert(0, str(bagel_root))
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.qwen2 import Qwen2Tokenizer
from torch.utils.data import DataLoader
from data.transforms import ImageResizeTransform

# Import local ThinkTrace dataset
from data.interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset


def create_thinktrace_config():
    """Create a config for ThinkTrace dataset."""
    config = {
        'think_trace': {
            'dataset_names': ['chess_thinktrace'],
            'jsonl_paths': ['./datasets/chess_thinktrace.jsonl'],
            'data_dirs': ['./datasets/'],
            'image_prefix_dir': './datasets/chess_thinktrace_images',
            'num_used_data': [3],  # Use all samples
            'shuffle_lines': False,
            'shuffle_seed': 42,
            'image_transform_args': {
                'image_stride': 16,
                'max_image_size': 1024,
                'min_image_size': 512,
            },
            'vit_image_transform_args': {
                'image_stride': 14,
                'max_image_size': 980,
                'min_image_size': 378,
                'max_pixels': 2_007_040,
            },
            'is_mandatory': True,
            'weight': 1.0,
        }
    }
    return config


def visualize_thinktrace_sample(sample, tokenizer):
    """Visualize a single ThinkTrace sample."""
    print("\n" + "="*80)
    print("THINKTRACE SAMPLE VISUALIZATION")
    print("="*80)
    
    # Show the sequence plan
    print("\n📋 SEQUENCE PLAN:")
    print(f"  Total items in sequence: {len(sample['sequence_plan'])}")
    
    for i, item in enumerate(sample['sequence_plan']):
        item_type = item['type']
        has_loss = item.get('loss', 0)
        enable_cfg = item.get('enable_cfg', 0)
        
        if item_type == 'text':
            text_idx = len([x for x in sample['sequence_plan'][:i] if x['type'] == 'text'])
            if text_idx < len(sample['text_ids_list']):
                text_ids = sample['text_ids_list'][text_idx]
                text = tokenizer.decode(text_ids)
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"  [{i:3}] TEXT (loss={has_loss}, cfg={enable_cfg}): {preview}")
        elif item_type == 'vae_image':
            print(f"  [{i:3}] VAE_IMAGE (loss={has_loss}, cfg={enable_cfg})")
        elif item_type == 'vit_image':
            print(f"  [{i:3}] VIT_IMAGE (loss={has_loss}, cfg={enable_cfg})")
    
    print(f"\n📊 STATISTICS:")
    print(f"  Total tokens: {sample['num_tokens']}")
    print(f"  Text segments: {len(sample['text_ids_list'])}")
    print(f"  Image tensors: {len(sample['image_tensor_list'])}")
    
    # Show text content
    print(f"\n📝 TEXT CONTENT:")
    for i, text_ids in enumerate(sample['text_ids_list']):
        text = tokenizer.decode(text_ids)
        if '<think>' in text:
            print(f"  [Segment {i}] THINKING: {text[:100]}...")
        elif '<answer>' in text:
            print(f"  [Segment {i}] ANSWER: {text[:100]}...")
        else:
            print(f"  [Segment {i}]: {text[:100]}...")
    
    # Show how losses are applied
    print(f"\n🎯 LOSS CONFIGURATION:")
    text_with_loss = sum(1 for item in sample['sequence_plan'] if item['type'] == 'text' and item.get('loss', 0))
    vae_with_loss = sum(1 for item in sample['sequence_plan'] if item['type'] == 'vae_image' and item.get('loss', 0))
    print(f"  Text segments with CE loss: {text_with_loss}")
    print(f"  VAE images with MSE loss: {vae_with_loss}")
    
    return sample


def test_thinktrace_dataset():
    """Test the ThinkTrace dataset loading."""
    print("\n" + "="*80)
    print("TESTING THINKTRACE DATASET")
    print("="*80)
    
    # Initialize tokenizer
    tokenizer_path = "/home/colligo/project/vlm/Bagel/hf/Qwen2.5-0.5B-Instruct/"
    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer = add_special_tokens(tokenizer)
    
    # Create transforms
    vae_transform = ImageResizeTransform(
        image_stride=16,
        max_image_size=1024,
        min_image_size=512,
    )
    
    vit_transform = ImageResizeTransform(
        image_stride=14,
        max_image_size=980,
        min_image_size=378,
        max_pixels=2_007_040,
    )
    
    # Create dataset
    dataset = ThinkTraceJSONLIterableDataset(
        dataset_name="chess_thinktrace",
        transform=vae_transform,
        tokenizer=tokenizer,
        vit_transform=vit_transform,
        jsonl_path_list=["./datasets/chess_thinktrace.jsonl"],
        data_dir_list=["./datasets/"],
        num_used_data=[3],
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=42,
        image_prefix_dir="./datasets/chess_thinktrace_images",
    )
    
    print("\nDataset created successfully!")
    print(f"  Dataset name: {dataset.dataset_name}")
    print(f"  Data paths: {len(dataset.data_paths)} samples")
    
    # Get first sample
    print("\nFetching first sample...")
    iterator = iter(dataset)
    
    try:
        sample = next(iterator)
        visualize_thinktrace_sample(sample, tokenizer)
    except StopIteration:
        print("No samples found in dataset")
    except Exception as e:
        print(f"Error loading sample: {e}")
        import traceback
        traceback.print_exc()


def test_packed_dataset():
    """Test the packed dataset with ThinkTrace data."""
    print("\n" + "="*80)
    print("TESTING PACKED DATASET WITH THINKTRACE")
    print("="*80)
    
    # Initialize tokenizer
    tokenizer_path = "/home/colligo/project/vlm/Bagel/hf/Qwen2.5-0.5B-Instruct/"
    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer = add_special_tokens(tokenizer)
    
    # Create data config
    data_config = DataConfig(
        data_type='think_trace',
        tokenizer=tokenizer,
        max_num_tokens=2048,  # Small for testing
        expected_num_tokens=1024,
        vae_image_downsample=16,
        vit_patch_size=14,
        max_latent_size=32,
        max_num_patch_per_side=70,
        text_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vit_cond_dropout_prob=0.0,
        ce_loss_reweighting=False,
    )
    
    # Create transforms
    vae_transform = ImageResizeTransform(
        image_stride=16,
        max_image_size=1024,
        min_image_size=512,
    )
    
    vit_transform = ImageResizeTransform(
        image_stride=14,
        max_image_size=980,
        min_image_size=378,
        max_pixels=2_007_040,
    )
    
    # Create ThinkTrace dataset
    thinktrace_dataset = ThinkTraceJSONLIterableDataset(
        dataset_name="chess_thinktrace",
        transform=vae_transform,
        tokenizer=tokenizer,
        vit_transform=vit_transform,
        jsonl_path_list=["./datasets/chess_thinktrace.jsonl"],
        data_dir_list=["./datasets/"],
        num_used_data=[3],
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=42,
        image_prefix_dir="./datasets/chess_thinktrace_images",
    )
    
    # Create packed dataset
    packed_dataset = PackedDataset(
        data_config=data_config,
        dataset_configs={'think_trace': {'weight': 1.0, 'is_mandatory': True}},
        dataset_groups={'think_trace': [thinktrace_dataset]},
        local_rank=0,
        world_size=1,
        num_workers=1,
    )
    
    print("\nPacked dataset created!")
    print(f"  Max tokens per sequence: {data_config.max_num_tokens}")
    print(f"  Expected tokens: {data_config.expected_num_tokens}")
    
    # Create dataloader
    dataloader = DataLoader(
        packed_dataset,
        batch_size=1,
        collate_fn=collate_wrapper,
        num_workers=0,
    )
    
    print("\nFetching first batch...")
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # Just show first batch
                break
            
            print(f"\n📦 BATCH {batch_idx} STRUCTURE:")
            print(f"  Sequence length: {batch.sequence_length}")
            print(f"  Packed samples: {len(batch.sample_lens)}")
            print(f"  Sample lengths: {batch.sample_lens}")
            
            # Show token distribution
            print(f"\n  Token Distribution:")
            print(f"    Text tokens: {len(batch.packed_text_ids)}")
            if hasattr(batch, 'packed_vit_tokens'):
                print(f"    VIT tokens: {len(batch.packed_vit_tokens) if len(batch.packed_vit_tokens) > 0 else 0}")
            if hasattr(batch, 'packed_vae_token_indexes'):
                print(f"    VAE token positions: {len(batch.packed_vae_token_indexes)}")
            
            # Show loss indexes
            if hasattr(batch, 'ce_loss_indexes'):
                print(f"\n  Loss Configuration:")
                print(f"    CE loss positions: {len(batch.ce_loss_indexes)} tokens")
            if hasattr(batch, 'mse_loss_indexes'):
                print(f"    MSE loss positions: {len(batch.mse_loss_indexes)} tokens")
            
            # Decode some text
            print(f"\n  Sample Text (first 100 tokens):")
            text_preview = tokenizer.decode(batch.packed_text_ids[:100].tolist())
            print(f"    {text_preview[:200]}...")
            
    except Exception as e:
        print(f"Error in dataloader: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Visualize ThinkTrace dataset and dataloader")
    parser.add_argument("--mode", choices=["dataset", "packed", "both"], default="both",
                        help="What to test: dataset only, packed dataset, or both")
    parser.add_argument("--convert-first", action="store_true",
                        help="Run conversion script first")
    
    args = parser.parse_args()
    
    # Run conversion if requested
    if args.convert_first:
        print("Running conversion script first...")
        import subprocess
        result = subprocess.run([sys.executable, "convert_zebra_to_thinktrace.py"], 
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Conversion failed: {result.stderr}")
            return
        print("Conversion complete!\n")
    
    # Check if JSONL exists
    jsonl_path = Path("./datasets/chess_thinktrace.jsonl")
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found!")
        print("Run with --convert-first flag to create it first")
        return
    
    # Run tests
    if args.mode in ["dataset", "both"]:
        test_thinktrace_dataset()
    
    if args.mode in ["packed", "both"]:
        test_packed_dataset()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nThis demonstrates how ThinkTrace dataset works:")
    print("1. Questions can contain images via <image_start>[key]<image_end>")
    print("2. Reasoning is wrapped in <think></think> tokens")
    print("3. Answers are wrapped in <answer></answer> tokens")
    print("4. Images can be generated during reasoning (MSE loss)")
    print("5. Text generation uses CE loss")
    print("\nThe model learns to:")
    print("- Understand questions with images (VIT)")
    print("- Generate step-by-step reasoning")
    print("- Create visual aids during reasoning (VAE)")
    print("- Provide clear final answers")


if __name__ == "__main__":
    main()