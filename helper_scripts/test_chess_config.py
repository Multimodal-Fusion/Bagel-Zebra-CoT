#!/usr/bin/env python3
"""
Test loading chess ThinkTrace dataset configuration.
Simpler version without full PackedDataset to avoid import issues.
"""

import sys
import json
from pathlib import Path

# Setup paths
bagel_root = Path("/home/colligo/project/vlm/Bagel")
zebra_root = Path("/home/colligo/project/vlm/Bagel-Zebra-CoT")
sys.path.insert(0, str(bagel_root))
sys.path.insert(0, str(zebra_root))

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.qwen2 import Qwen2Tokenizer
from data.interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset


def test_dataset_loading():
    """Test that we can load the chess dataset with proper configuration."""
    
    print("\n" + "="*80)
    print("TESTING CHESS THINKTRACE CONFIGURATION")
    print("="*80)
    
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer, special_tokens, num_new_tokens = add_special_tokens(tokenizer)
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
    
    # Check dataset info exists
    from data.dataset_info import DATASET_INFO
    if 'think_trace' in DATASET_INFO and 'chess_thinktrace' in DATASET_INFO['think_trace']:
        info = DATASET_INFO['think_trace']['chess_thinktrace']
        print("\n✅ Dataset info found in DATASET_INFO:")
        print(f"  JSONL path: {info['jsonl_path']}")
        print(f"  Image prefix: {info['image_prefix_dir']}")
        print(f"  Data dir: {info['data_dir']}")
        print(f"  Num samples: {info['num_total_samples']}")
    else:
        print("❌ Dataset info not found in DATASET_INFO")
        return
    
    # Create dataset using the info
    dataset = ThinkTraceJSONLIterableDataset(
        dataset_name="chess_thinktrace",
        transform=vae_transform,
        tokenizer=tokenizer,
        vit_transform=vit_transform,
        jsonl_path_list=[info['jsonl_path']],
        data_dir_list=[info['data_dir']],
        num_used_data=[info['num_total_samples']],
        local_rank=0,
        world_size=1,
        num_workers=1,
        shuffle_lines=True,
        shuffle_seed=42,
        image_prefix_dir=info['image_prefix_dir'],
    )
    
    print("\n✅ Dataset created successfully")
    
    # Test loading samples
    print("\n📊 Testing sample loading...")
    for i, sample in enumerate(dataset):
        if i >= 3:  # Test first 3 samples
            break
        
        print(f"\n  Sample {i + 1}:")
        print(f"    Total tokens: {sample['num_tokens']}")
        print(f"    Text segments: {len(sample['text_ids_list'])}")
        print(f"    Image tensors: {len(sample['image_tensor_list'])}")
        print(f"    Sequence plan items: {len(sample['sequence_plan'])}")
        
        # Count loss items
        text_with_loss = sum(1 for item in sample['sequence_plan'] 
                           if item['type'] == 'text' and item.get('loss', 0))
        vae_with_loss = sum(1 for item in sample['sequence_plan'] 
                          if item['type'] == 'vae_image' and item.get('loss', 0))
        vit_count = sum(1 for item in sample['sequence_plan'] 
                       if item['type'] == 'vit_image')
        
        print(f"    Text segments with CE loss: {text_with_loss}")
        print(f"    VAE images with MSE loss: {vae_with_loss}")
        print(f"    VIT images (understanding): {vit_count}")
    
    print("\n" + "="*80)
    print("CONFIGURATION TEST COMPLETE")
    print("="*80)
    
    print("\n📝 Training Configuration Summary:")
    print("  Dataset: chess_thinktrace (10 samples)")
    print("  Features:")
    print("    - Chain-of-thought reasoning with <think> tokens")
    print("    - Visual reasoning with generated images")
    print("    - Interleaved text (CE loss) and images (MSE loss)")
    print("    - Triple image representation (VAE loss, VAE cond, VIT)")
    
    print("\n🎯 Next Steps for Training:")
    print("  1. Use PackedDataset to pack multiple samples efficiently")
    print("  2. Configure optimizer for dual loss (CE + MSE)")
    print("  3. Set up model with both text and image generation heads")
    print("  4. Implement training loop with proper loss weighting")


if __name__ == "__main__":
    test_dataset_loading()