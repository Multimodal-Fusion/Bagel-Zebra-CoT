#!/bin/bash
# Debug script for chess ThinkTrace dataset
# Tests data loading without full training

# Environment setup
export HF_HOME=/home/colligo/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/colligo/project/vlm/Bagel:/home/colligo/project/vlm/Bagel-Zebra-CoT:$PYTHONPATH

echo "=========================================="
echo "Debug Mode: Chess ThinkTrace Dataset"
echo "=========================================="

# Test data loading only
python3 << 'EOF'
import sys
from pathlib import Path

# Setup paths
bagel_root = Path("/home/colligo/project/vlm/Bagel")
zebra_root = Path("/home/colligo/project/vlm/Bagel-Zebra-CoT")
sys.path.insert(0, str(bagel_root))
sys.path.insert(0, str(zebra_root))

import torch
import yaml
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.qwen2 import Qwen2Tokenizer
from data.dataset_info import DATASET_INFO, DATASET_REGISTRY

print("\n1. Loading tokenizer...")
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer, special_tokens, num_new_tokens = add_special_tokens(tokenizer)
print(f"   ✅ Tokenizer loaded with {num_new_tokens} new tokens")

print("\n2. Loading dataset config...")
with open("./data/configs/chess_thinktrace.yaml", 'r') as f:
    config = yaml.safe_load(f)
print("   ✅ Config loaded:", list(config.keys()))

print("\n3. Checking dataset registry...")
if 'think_trace' in DATASET_REGISTRY:
    print(f"   ✅ Dataset class found: {DATASET_REGISTRY['think_trace'].__name__}")
else:
    print("   ❌ Dataset class not found in registry")

print("\n4. Checking dataset info...")
if 'think_trace' in DATASET_INFO and 'chess_thinktrace' in DATASET_INFO['think_trace']:
    info = DATASET_INFO['think_trace']['chess_thinktrace']
    print(f"   ✅ Dataset info found:")
    print(f"      - JSONL: {info['jsonl_path']}")
    print(f"      - Images: {info['image_prefix_dir']}")
    print(f"      - Samples: {info['num_total_samples']}")
else:
    print("   ❌ Dataset info not found")

print("\n5. Creating dataset instance...")
try:
    dataset_class = DATASET_REGISTRY['think_trace']
    dataset_config = config['think_trace']
    
    # Create transforms
    vae_transform = ImageTransform(**dataset_config['image_transform_args'])
    vit_transform = ImageTransform(**dataset_config['vit_image_transform_args'])
    
    # Get dataset info
    dataset_name = dataset_config['dataset_names'][0]
    dataset_info = DATASET_INFO['think_trace'][dataset_name]
    
    # Create dataset
    dataset = dataset_class(
        dataset_name=dataset_name,
        transform=vae_transform,
        tokenizer=tokenizer,
        vit_transform=vit_transform,
        jsonl_path_list=[dataset_info['jsonl_path']],
        data_dir_list=[dataset_info['data_dir']],
        num_used_data=dataset_config['num_used_data'],
        local_rank=0,
        world_size=1,
        num_workers=1,
        shuffle_lines=dataset_config.get('shuffle_lines', True),
        shuffle_seed=dataset_config.get('shuffle_seed', 42),
        image_prefix_dir=dataset_info.get('image_prefix_dir'),
    )
    print("   ✅ Dataset created successfully")
    
    print("\n6. Testing sample loading...")
    for i, sample in enumerate(dataset):
        if i >= 1:
            break
        print(f"   ✅ Sample loaded:")
        print(f"      - Tokens: {sample['num_tokens']}")
        print(f"      - Text segments: {len(sample['text_ids_list'])}")
        print(f"      - Images: {len(sample['image_tensor_list'])}")
        
        # Count losses
        text_loss = sum(1 for item in sample['sequence_plan'] 
                       if item['type'] == 'text' and item.get('loss', 0))
        vae_loss = sum(1 for item in sample['sequence_plan'] 
                      if item['type'] == 'vae_image' and item.get('loss', 0))
        print(f"      - CE loss items: {text_loss}")
        print(f"      - MSE loss items: {vae_loss}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n========================================")
print("Debug complete!")
print("========================================")
EOF

echo ""
echo "To run full training, use: ./scripts/train_chess_thinktrace.sh"