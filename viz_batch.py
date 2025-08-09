#!/usr/bin/env python3
"""
Visualize what's actually in a batch from the ThinkTrace dataloader.
Shows vision tokens, text tokens, positions, loss masks, etc.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add paths
bagel_root = Path("/home/colligo/project/vlm/Bagel")
sys.path.insert(0, str(bagel_root))
sys.path.insert(0, str(Path(__file__).parent))

# Import from Bagel (use their modules)
sys.path.insert(0, str(bagel_root))
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.qwen2 import Qwen2Tokenizer
from torch.utils.data import DataLoader
from data.transforms import ImageResizeTransform

# Import local ThinkTrace
sys.path.insert(0, str(Path(__file__).parent))
from data.interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset


def show_batch_contents(batch, tokenizer):
    """Show exactly what's in a batch."""
    
    print("\n" + "="*80)
    print("BATCH CONTENTS")
    print("="*80)
    
    # Basic info
    print(f"\n📏 DIMENSIONS:")
    print(f"  sequence_length: {batch.sequence_length}")
    print(f"  batch_size: 1 (always)")
    print(f"  samples packed: {len(batch.sample_lens)}")
    print(f"  sample lengths: {batch.sample_lens}")
    
    # Text tokens
    print(f"\n📝 TEXT TOKENS:")
    print(f"  packed_text_ids shape: {batch.packed_text_ids.shape}")
    print(f"  packed_text_indexes shape: {batch.packed_text_indexes.shape}")
    print(f"  Total text tokens: {len(batch.packed_text_ids)}")
    
    # Show first few text tokens
    print(f"\n  First 20 text token IDs: {batch.packed_text_ids[:20].tolist()}")
    text_preview = tokenizer.decode(batch.packed_text_ids[:50].tolist())
    print(f"  Decoded: '{text_preview}'")
    
    # Text positions in sequence
    print(f"\n  Text token positions (first 20): {batch.packed_text_indexes[:20].tolist()}")
    
    # Vision tokens - VIT
    if hasattr(batch, 'packed_vit_tokens') and batch.packed_vit_tokens is not None:
        print(f"\n👁️ VIT TOKENS (for understanding):")
        print(f"  packed_vit_tokens shape: {batch.packed_vit_tokens.shape}")
        print(f"  packed_vit_token_indexes shape: {batch.packed_vit_token_indexes.shape}")
        print(f"  vit_token_seqlens: {batch.vit_token_seqlens.tolist()}")
        
        # VIT positions
        print(f"  VIT token positions (first 20): {batch.packed_vit_token_indexes[:20].tolist()}")
    
    # Vision tokens - VAE
    if hasattr(batch, 'packed_vae_token_indexes') and len(batch.packed_vae_token_indexes) > 0:
        print(f"\n🎨 VAE TOKENS (for generation):")
        print(f"  packed_vae_token_indexes shape: {batch.packed_vae_token_indexes.shape}")
        print(f"  VAE token positions (first 20): {batch.packed_vae_token_indexes[:20].tolist()}")
        
        if hasattr(batch, 'padded_latent'):
            print(f"  padded_latent shape: {batch.padded_latent.shape if batch.padded_latent is not None else 'None'}")
    
    # Position IDs (RoPE)
    print(f"\n🔢 POSITION IDS (RoPE):")
    print(f"  packed_position_ids shape: {batch.packed_position_ids.shape}")
    print(f"  First 30 position IDs: {batch.packed_position_ids[:30].tolist()}")
    
    # Loss masks
    print(f"\n🎯 LOSS MASKS:")
    if hasattr(batch, 'ce_loss_indexes'):
        print(f"  CE loss indexes (text): {batch.ce_loss_indexes.shape}")
        print(f"    First 20 CE positions: {batch.ce_loss_indexes[:20].tolist()}")
        print(f"    Total CE tokens: {len(batch.ce_loss_indexes)}")
    
    if hasattr(batch, 'mse_loss_indexes'):
        print(f"  MSE loss indexes (images): {batch.mse_loss_indexes.shape}")
        if len(batch.mse_loss_indexes) > 0:
            print(f"    First 20 MSE positions: {batch.mse_loss_indexes[:20].tolist()}")
            print(f"    Total MSE tokens: {len(batch.mse_loss_indexes)}")
    
    # Labels
    if hasattr(batch, 'packed_label_ids'):
        print(f"\n🏷️ LABELS:")
        print(f"  packed_label_ids shape: {batch.packed_label_ids.shape}")
        print(f"  First 20 label IDs: {batch.packed_label_ids[:20].tolist()}")
    
    # Visualize sequence structure
    print(f"\n🗺️ SEQUENCE MAP (first 200 positions):")
    sequence_map = ['_'] * min(200, batch.sequence_length)
    
    # Mark text positions
    for idx in batch.packed_text_indexes[:200].tolist():
        if idx < len(sequence_map):
            sequence_map[idx] = 'T'
    
    # Mark VIT positions
    if hasattr(batch, 'packed_vit_token_indexes'):
        for idx in batch.packed_vit_token_indexes[:200].tolist():
            if idx < len(sequence_map):
                sequence_map[idx] = 'V'
    
    # Mark VAE positions
    if hasattr(batch, 'packed_vae_token_indexes'):
        for idx in batch.packed_vae_token_indexes[:200].tolist():
            if idx < len(sequence_map):
                sequence_map[idx] = 'G'  # G for Generation
    
    print("  T=Text, V=VIT(understanding), G=VAE(generation), _=Empty")
    for i in range(0, len(sequence_map), 50):
        chunk = sequence_map[i:i+50]
        print(f"  [{i:3d}-{i+49:3d}]: {''.join(chunk)}")
    
    # Show what's being trained
    print(f"\n⚡ TRAINING SIGNALS:")
    ce_count = len(batch.ce_loss_indexes) if hasattr(batch, 'ce_loss_indexes') else 0
    mse_count = len(batch.mse_loss_indexes) if hasattr(batch, 'mse_loss_indexes') else 0
    
    print(f"  CE loss applied to {ce_count} text tokens")
    print(f"  MSE loss applied to {mse_count} image tokens")
    
    # Special tokens
    print(f"\n🔤 SPECIAL TOKENS:")
    special_ids = {
        'vision_start': tokenizer.convert_tokens_to_ids('<|vision_start|>'),
        'vision_end': tokenizer.convert_tokens_to_ids('<|vision_end|>'),
        'im_start': tokenizer.convert_tokens_to_ids('<|im_start|>'),
        'im_end': tokenizer.convert_tokens_to_ids('<|im_end|>'),
    }
    
    for name, token_id in special_ids.items():
        count = (batch.packed_text_ids == token_id).sum().item()
        print(f"  {name} ({token_id}): appears {count} times")


def main():
    print("\n" + "="*80)
    print("THINKTRACE BATCH VISUALIZATION")
    print("="*80)
    
    # Initialize tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained("/home/colligo/project/vlm/Bagel/hf/Qwen2.5-0.5B-Instruct/")
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
    
    # Create data config
    data_config = DataConfig(
        data_type='think_trace',
        tokenizer=tokenizer,
        max_num_tokens=4096,
        expected_num_tokens=2048,
        vae_image_downsample=16,
        vit_patch_size=14,
        max_latent_size=32,
        max_num_patch_per_side=70,
        text_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vit_cond_dropout_prob=0.0,
        ce_loss_reweighting=False,
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
        shuffle_lines=False,
        image_prefix_dir="./datasets/chess_thinktrace_images",
    )
    
    # Create packed dataset
    packed_dataset = PackedDataset(
        data_config=data_config,
        dataset_configs={'think_trace': {'weight': 1.0, 'is_mandatory': True}},
        dataset_groups={'think_trace': [dataset]},
        local_rank=0,
        world_size=1,
        num_workers=1,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        packed_dataset,
        batch_size=1,
        collate_fn=collate_wrapper,
        num_workers=0,
    )
    
    # Get first batch
    print("\nLoading batch...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:
            break
        
        show_batch_contents(batch, tokenizer)
        
        print("\n" + "="*80)
        print("END OF BATCH")
        print("="*80)


if __name__ == "__main__":
    main()