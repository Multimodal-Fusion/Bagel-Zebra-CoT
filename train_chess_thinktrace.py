#!/usr/bin/env python3
"""
Training script for chess ThinkTrace dataset with interleaved text and image generation.
"""

import sys
from pathlib import Path

# Setup paths
bagel_root = Path("/home/colligo/project/vlm/Bagel")
zebra_root = Path("/home/colligo/project/vlm/Bagel-Zebra-CoT")
sys.path.insert(0, str(bagel_root))
sys.path.insert(0, str(zebra_root))

import torch
from torch.utils.data import DataLoader
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.qwen2 import Qwen2Tokenizer


def create_data_config():
    """Create data configuration for training."""
    grouped_datasets = {
        'think_trace': {
            'dataset_names': ['chess_thinktrace'],
            'weight': 1.0,
            'is_mandatory': True,
            'num_used_data': [10],  # Use all 10 samples from toy dataset
            'shuffle_lines': True,
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
        }
    }
    
    data_config = DataConfig(
        grouped_datasets=grouped_datasets,
        text_cond_dropout_prob=0.1,  # 10% dropout for CFG during training
        vae_cond_dropout_prob=0.1,   # 10% dropout for VAE CFG
        vit_cond_dropout_prob=0.4,   # 40% dropout for VIT CFG (higher as per original)
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    )
    
    return data_config


def main():
    print("\n" + "="*80)
    print("CHESS THINKTRACE TRAINING CONFIGURATION")
    print("="*80)
    
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer, special_tokens, num_new_tokens = add_special_tokens(tokenizer)
    print(f"✅ Tokenizer loaded with {num_new_tokens} new tokens")
    
    # Create data config
    data_config = create_data_config()
    print("\n📊 Data Configuration:")
    print(f"  Grouped datasets: {list(data_config.grouped_datasets.keys())}")
    print(f"  Text CFG dropout: {data_config.text_cond_dropout_prob}")
    print(f"  VAE CFG dropout: {data_config.vae_cond_dropout_prob}")
    print(f"  VIT CFG dropout: {data_config.vit_cond_dropout_prob}")
    
    # Create packed dataset
    packed_dataset = PackedDataset(
        data_config=data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=0,
        world_size=1,
        num_workers=4,
        expected_num_tokens=8192,  # Expect ~8k tokens per batch
        max_num_tokens=16384,      # Max 16k tokens per batch
        max_num_tokens_per_sample=8192,  # Max 8k per sample
        prefer_buffer_before=4096,
        max_buffer_size=10,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    )
    
    print("\n📦 Packed Dataset Configuration:")
    print(f"  Expected tokens per batch: {packed_dataset.expected_num_tokens}")
    print(f"  Max tokens per batch: {packed_dataset.max_num_tokens}")
    print(f"  Max tokens per sample: {packed_dataset.max_num_tokens_per_sample}")
    
    # Create dataloader
    dataloader = DataLoader(
        packed_dataset,
        batch_size=1,  # Always 1 for packed dataset
        collate_fn=collate_wrapper(),
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True,
    )
    
    print("\n🎯 Training Features:")
    print("  1. Interleaved text and image generation")
    print("  2. Chain-of-thought reasoning with <think> tokens")
    print("  3. Visual reasoning with generated images")
    print("  4. Dual loss: CE for text, MSE for images")
    print("  5. Triple image representation: VAE loss, VAE conditioning, VIT understanding")
    
    # Test loading a batch
    print("\n🧪 Testing batch loading...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:
            break
        
        print(f"\n✅ Successfully loaded batch {batch_idx + 1}")
        print(f"  Sequence length: {batch.sequence_length}")
        print(f"  Samples packed: {len(batch.sample_lens)}")
        print(f"  Sample lengths: {batch.sample_lens}")
        
        # Check losses
        ce_count = len(batch.ce_loss_indexes) if hasattr(batch, 'ce_loss_indexes') else 0
        mse_count = len(batch.mse_loss_indexes) if hasattr(batch, 'mse_loss_indexes') else 0
        
        print(f"\n  Loss configuration:")
        print(f"    CE loss on {ce_count} text tokens")
        print(f"    MSE loss on {mse_count} image tokens")
        
        # Check special tokens
        vision_start = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        vision_end = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        
        print(f"\n  Special tokens in batch:")
        print(f"    vision_start: {(batch.packed_text_ids == vision_start).sum().item()}")
        print(f"    vision_end: {(batch.packed_text_ids == vision_end).sum().item()}")
        print(f"    im_start: {(batch.packed_text_ids == im_start).sum().item()}")
        print(f"    im_end: {(batch.packed_text_ids == im_end).sum().item()}")
    
    print("\n" + "="*80)
    print("CONFIGURATION READY FOR TRAINING")
    print("="*80)
    print("\nTo use this in training:")
    print("1. Import this configuration in your training script")
    print("2. Pass the dataloader to your training loop")
    print("3. Compute CE loss on batch.ce_loss_indexes positions")
    print("4. Compute MSE loss on batch.mse_loss_indexes positions")
    print("5. The model should handle both text generation and image generation")


if __name__ == "__main__":
    main()