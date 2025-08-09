#!/usr/bin/env python3
"""
Simple test of ThinkTrace dataset batch without PackedDataset complexity
"""

import sys
from pathlib import Path

# Setup paths
bagel_root = Path("/home/colligo/project/vlm/Bagel")
zebra_root = Path("/home/colligo/project/vlm/Bagel-Zebra-CoT")
sys.path.insert(0, str(bagel_root))
sys.path.insert(0, str(zebra_root))

import torch
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.qwen2 import Qwen2Tokenizer
from data.interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset


def main():
    print("\n" + "="*80)
    print("THINKTRACE DATASET - RAW BATCH VISUALIZATION")
    print("="*80)
    
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer, special_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    print(f"✅ Tokenizer loaded with {num_new_tokens} new tokens")
    
    # Create transforms
    vae_transform = ImageTransform(
        image_stride=16,
        max_image_size=1024,
        min_image_size=512,
    )
    
    vit_transform = ImageTransform(
        image_stride=14,
        max_image_size=980,
        min_image_size=378,
        max_pixels=2_007_040,
    )
    
    # Create ThinkTrace dataset
    dataset = ThinkTraceJSONLIterableDataset(
        dataset_name="chess_thinktrace",
        transform=vae_transform,
        tokenizer=tokenizer,
        vit_transform=vit_transform,
        jsonl_path_list=[str(zebra_root / "datasets/chess_thinktrace.jsonl")],
        data_dir_list=[str(zebra_root / "datasets/")],
        num_used_data=[3],
        local_rank=0,
        world_size=1,
        num_workers=1,
        shuffle_lines=False,
        image_prefix_dir=str(zebra_root / "datasets/chess_thinktrace_images"),
    )
    
    print("\n📊 Getting first sample from dataset...")
    
    # Get first sample
    for i, sample in enumerate(dataset):
        if i >= 1:  # Just first sample
            break
        
        print(f"\n📦 SAMPLE STRUCTURE:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Total tokens: {sample['num_tokens']}")
        print(f"  Text segments: {len(sample['text_ids_list'])}")
        print(f"  Image tensors: {len(sample['image_tensor_list'])}")
        
        # Show sequence plan
        print(f"\n📋 SEQUENCE PLAN ({len(sample['sequence_plan'])} items):")
        for idx, item in enumerate(sample['sequence_plan'][:10]):  # First 10
            item_type = item['type']
            has_loss = item.get('loss', 0)
            enable_cfg = item.get('enable_cfg', 0)
            print(f"  [{idx:2}] {item_type:12} loss={has_loss} cfg={enable_cfg}")
        if len(sample['sequence_plan']) > 10:
            print(f"  ... {len(sample['sequence_plan']) - 10} more items")
        
        # Decode text segments
        print(f"\n📝 TEXT SEGMENTS:")
        for i, text_ids in enumerate(sample['text_ids_list'][:5]):  # First 5
            text = tokenizer.decode(text_ids)
            preview = text[:100].replace('\n', ' ')
            if '<think>' in text:
                print(f"  [{i}] THINKING: {preview}...")
            elif '<answer>' in text:
                print(f"  [{i}] ANSWER: {preview}...")
            elif 'Question:' in text:
                print(f"  [{i}] QUESTION: {preview}...")
            else:
                print(f"  [{i}] TEXT: {preview}...")
        
        # Show image info
        print(f"\n🖼️ IMAGE TENSORS:")
        for i, img_tensor in enumerate(sample['image_tensor_list'][:5]):
            print(f"  [{i}] Shape: {img_tensor.shape}")
        
        # Count losses
        text_with_loss = sum(1 for item in sample['sequence_plan'] if item['type'] == 'text' and item.get('loss', 0))
        vae_with_loss = sum(1 for item in sample['sequence_plan'] if item['type'] == 'vae_image' and item.get('loss', 0))
        vit_count = sum(1 for item in sample['sequence_plan'] if item['type'] == 'vit_image')
        
        print(f"\n🎯 LOSS CONFIGURATION:")
        print(f"  Text segments with CE loss: {text_with_loss}")
        print(f"  VAE images with MSE loss: {vae_with_loss}")
        print(f"  VIT images (understanding): {vit_count}")
        
        # Show how it would be packed
        print(f"\n📦 WHEN PACKED INTO BATCH:")
        print(f"  This creates an interleaved sequence of:")
        print(f"  1. Question with problem images (VIT for understanding)")
        print(f"  2. Thinking text with <think> tokens (CE loss)")
        print(f"  3. Generated reasoning images (VAE with MSE loss)")
        print(f"  4. Final answer with <answer> tokens (CE loss)")
        print(f"\n  Total sequence would be ~{sample['num_tokens']} tokens")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()