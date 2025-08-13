#!/usr/bin/env python3
"""Standalone script to test textual-cot data loading"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import yaml
from data.vlm_dataset import SftJSONLIterableDataset
from transformers import AutoTokenizer

# Mock classes for testing
class MockTransform:
    def __init__(self):
        self.stride = 16
    
    def __call__(self, image, img_num=1):
        import torch
        return torch.randn(3, 512, 512)

class MockFrameSampler:
    def __call__(self, video_path):
        return []

def test_textual_cot_loading():
    """Test loading textual-cot dataset"""
    
    print("="*60)
    print("Testing Textual CoT Dataset Loading")
    print("="*60)
    
    # Load the config to get parameters
    config_path = '/home/colligo/project/vlm/Bagel-Zebra-CoT/data/configs/frozenlake/textual-cot.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfig loaded from: {config_path}")
    print(f"Config keys: {list(config.keys())}")
    
    textual_cot_config = config.get('textual_cot', {})
    print(f"\nTextual CoT config:")
    for key, value in textual_cot_config.items():
        if key != 'image_transform_args' and key != 'frame_sampler_args':
            print(f"  {key}: {value}")
    
    # Get dataset info
    from data.dataset_info import DATASET_INFO
    dataset_info = DATASET_INFO.get('textual_cot', {}).get('frozenlake_textual_cot', {})
    print(f"\nDataset info from DATASET_INFO:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # Prepare parameters
    jsonl_path = dataset_info.get('jsonl_path')
    data_dir = dataset_info.get('data_dir')
    use_thinktrace = dataset_info.get('use_thinktrace', False)
    trace_field = dataset_info.get('trace_field', 'Text Reasoning Trace[textual-cot]')
    num_used_data = textual_cot_config.get('num_used_data', [None])
    
    print(f"\nDataset parameters:")
    print(f"  jsonl_path: {jsonl_path}")
    print(f"  data_dir: {data_dir}")
    print(f"  use_thinktrace: {use_thinktrace}")
    print(f"  trace_field: {trace_field}")
    print(f"  num_used_data: {num_used_data}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = SftJSONLIterableDataset(
        dataset_name="frozenlake_textual_cot",
        transform=MockTransform(),
        tokenizer=tokenizer,
        frame_sampler=MockFrameSampler(),
        jsonl_path_list=[jsonl_path],
        data_dir_list=[data_dir],
        num_used_data=num_used_data,
        use_thinktrace=use_thinktrace,
        trace_field=trace_field,
        shuffle_lines=False,
    )
    
    print(f"\nDataset created successfully!")
    print(f"  use_thinktrace: {dataset.use_thinktrace}")
    print(f"  trace_field: {dataset.trace_field}")
    print(f"  Number of data paths: {len(dataset.data_paths)}")
    
    # Test the iterator
    print("\n" + "="*60)
    print("Testing data iteration (first 3 samples)...")
    print("="*60)
    
    data_iter = iter(dataset)
    
    for i in range(3):
        print(f"\n--- Sample {i+1} ---")
        try:
            # Get the raw data first to see what's happening
            if i < len(dataset.data_paths):
                raw_json, image_dir = dataset.data_paths[i]
                raw_data = json.loads(raw_json)
                print(f"Raw data keys: {list(raw_data.keys())[:10]}")
                print(f"Has Question: {'Question' in raw_data}")
                print(f"Has {trace_field}: {trace_field in raw_data}")
                print(f"Has Final Answer: {'Final Answer' in raw_data}")
                print(f"Has problem_image_1: {'problem_image_1' in raw_data}")
                
                # Test conversion directly
                print(f"\nTesting conversion with use_thinktrace={dataset.use_thinktrace}...")
                if dataset.use_thinktrace:
                    converted = dataset.convert_thinktrace_to_conversations(raw_data)
                    print(f"Converted keys: {list(converted.keys())}")
                    print(f"Has conversations: {'conversations' in converted}")
                    if 'conversations' in converted:
                        print(f"  Num conversations: {len(converted['conversations'])}")
                        print(f"  Human message length: {len(converted['conversations'][0]['value'])}")
                        print(f"  GPT message length: {len(converted['conversations'][1]['value'])}")
                    print(f"Has image: {'image' in converted}")
                    if 'image' in converted:
                        print(f"  Image path: {converted['image']}")
            
            # Now try to get from iterator
            print("\nGetting from iterator...")
            sample = next(data_iter)
            print(f"SUCCESS! Got sample from iterator")
            print(f"  Keys: {list(sample.keys())}")
            print(f"  Num text_ids_list: {len(sample.get('text_ids_list', []))}")
            print(f"  Num image_tensor_list: {len(sample.get('image_tensor_list', []))}")
            print(f"  Sequence plan length: {len(sample.get('sequence_plan', []))}")
            
            # Show sequence plan
            if 'sequence_plan' in sample:
                print(f"  Sequence plan:")
                for j, plan in enumerate(sample['sequence_plan'][:5]):
                    print(f"    {j}: type={plan['type']}, loss={plan['loss']}")
                if len(sample['sequence_plan']) > 5:
                    print(f"    ... and {len(sample['sequence_plan']) - 5} more")
                    
        except StopIteration:
            print("StopIteration - no more samples")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    test_textual_cot_loading()