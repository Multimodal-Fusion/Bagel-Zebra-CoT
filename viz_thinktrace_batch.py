#!/usr/bin/env python3
"""
Visualize ThinkTrace batch - standalone script that works
"""

import sys
from pathlib import Path

# Setup paths
bagel_root = Path("/home/colligo/project/vlm/Bagel")
zebra_root = Path("/home/colligo/project/vlm/Bagel-Zebra-CoT")
sys.path.insert(0, str(bagel_root))
sys.path.insert(0, str(zebra_root))

# Now import
import torch
from torch.utils.data import DataLoader
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
# Use correct class name
from data.transforms import ImageTransform as ImageResizeTransform
from modeling.qwen2 import Qwen2Tokenizer

# Import ThinkTrace from Zebra-CoT
from data.interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset


def main():
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer, special_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    print(f"✅ Tokenizer loaded with {num_new_tokens} new tokens")
    
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
    
    # Create data config with correct arguments
    grouped_datasets = {
        'think_trace': {
            'dataset_names': ['chess_thinktrace'],
            'weight': 1.0,
            'is_mandatory': True,
        }
    }
    
    data_config = DataConfig(
        grouped_datasets=grouped_datasets,
        text_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vit_cond_dropout_prob=0.0,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    )
    
    # Use the special tokens returned
    special_tokens = special_token_ids
    
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
    
    # Create packed dataset with correct arguments
    packed_dataset = PackedDataset(
        data_config=data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=0,
        world_size=1,
        num_workers=1,
        expected_num_tokens=2048,
        max_num_tokens=4096,
    )
    
    # Set the dataset groups manually
    packed_dataset.dataset_groups = {'think_trace': [dataset]}
    packed_dataset.dataset_configs = {'think_trace': {'weight': 1.0, 'is_mandatory': True}}
    
    # Create dataloader
    dataloader = DataLoader(
        packed_dataset,
        batch_size=1,
        collate_fn=collate_wrapper,
        num_workers=0,
    )
    
    print("\n" + "="*80)
    print("THINKTRACE BATCH VISUALIZATION")
    print("="*80)
    
    # Get first batch
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:
            break
        
        print(f"\n📦 BATCH STRUCTURE:")
        print(f"  sequence_length: {batch.sequence_length}")
        print(f"  samples packed: {len(batch.sample_lens)}")
        print(f"  sample lengths: {batch.sample_lens}")
        
        # Text tokens
        print(f"\n📝 TEXT TOKENS:")
        print(f"  Total: {len(batch.packed_text_ids)}")
        print(f"  First 30 IDs: {batch.packed_text_ids[:30].tolist()}")
        text_preview = tokenizer.decode(batch.packed_text_ids[:100].tolist())
        print(f"  Decoded preview: '{text_preview[:150]}...'")
        
        # VIT tokens (understanding)
        if hasattr(batch, 'packed_vit_tokens') and batch.packed_vit_tokens is not None:
            print(f"\n👁️ VIT TOKENS (understanding):")
            print(f"  Shape: {batch.packed_vit_tokens.shape}")
            print(f"  Positions (first 30): {batch.packed_vit_token_indexes[:30].tolist()}")
        
        # VAE tokens (generation)
        if hasattr(batch, 'packed_vae_token_indexes') and len(batch.packed_vae_token_indexes) > 0:
            print(f"\n🎨 VAE TOKENS (generation):")
            print(f"  Total positions: {len(batch.packed_vae_token_indexes)}")
            print(f"  Positions (first 30): {batch.packed_vae_token_indexes[:30].tolist()}")
        
        # Losses
        print(f"\n🎯 LOSSES:")
        if hasattr(batch, 'ce_loss_indexes'):
            print(f"  CE loss on {len(batch.ce_loss_indexes)} text tokens")
            print(f"    Positions (first 20): {batch.ce_loss_indexes[:20].tolist()}")
        if hasattr(batch, 'mse_loss_indexes'):
            print(f"  MSE loss on {len(batch.mse_loss_indexes)} image tokens")
            if len(batch.mse_loss_indexes) > 0:
                print(f"    Positions (first 20): {batch.mse_loss_indexes[:20].tolist()}")
        
        # Sequence map
        print(f"\n🗺️ SEQUENCE MAP (first 300 positions):")
        print("  T=Text, V=VIT, G=VAE/Generation, _=Empty")
        sequence_map = ['_'] * min(300, batch.sequence_length)
        
        for idx in batch.packed_text_indexes[:300].tolist():
            if idx < len(sequence_map):
                sequence_map[idx] = 'T'
        
        if hasattr(batch, 'packed_vit_token_indexes'):
            for idx in batch.packed_vit_token_indexes[:300].tolist():
                if idx < len(sequence_map):
                    sequence_map[idx] = 'V'
        
        if hasattr(batch, 'packed_vae_token_indexes'):
            for idx in batch.packed_vae_token_indexes[:300].tolist():
                if idx < len(sequence_map):
                    sequence_map[idx] = 'G'
        
        for i in range(0, len(sequence_map), 100):
            chunk = sequence_map[i:i+100]
            print(f"  [{i:3d}-{i+99:3d}]: {''.join(chunk)}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()