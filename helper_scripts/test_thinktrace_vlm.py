#!/usr/bin/env python3
"""Test script to verify thinktrace format works with SftJSONLIterableDataset"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from data.vlm_dataset import SftJSONLIterableDataset
from transformers import AutoTokenizer

# Mock classes for testing
class MockTransform:
    def __init__(self):
        self.stride = 16
    
    def __call__(self, image, img_num=1):
        import torch
        return torch.randn(3, 256, 256)

class MockFrameSampler:
    def __call__(self, video_path):
        return []

def test_thinktrace_conversion():
    """Test that thinktrace format is correctly converted to conversation format"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # Create dataset instance with thinktrace enabled
    dataset = SftJSONLIterableDataset(
        dataset_name="test_thinktrace",
        transform=MockTransform(),
        tokenizer=tokenizer,
        frame_sampler=MockFrameSampler(),
        jsonl_path_list=["/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/frozenlake/test/frozenlake_thinktrace.jsonl"],
        data_dir_list=["/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/"],
        num_used_data=[2],  # Test with just 2 samples
        use_thinktrace=True,
        trace_field='Text Reasoning Trace[textual-cot]',
        shuffle_lines=False,
    )
    
    # Test the conversion method directly
    sample_data = {
        "Question": "Test question with <image_start>[problem_image_1]<image_end>",
        "Text Reasoning Trace[textual-cot]": "This is my reasoning process",
        "Final Answer": "A",
        "problem_image_1": "data/frozenlake/test/images/sample_00000.png"
    }
    
    converted = dataset.convert_thinktrace_to_conversations(sample_data)
    
    print("Original data fields:", list(sample_data.keys()))
    print("\nConverted data has conversations:", 'conversations' in converted)
    print("Number of conversations:", len(converted.get('conversations', [])))
    
    if 'conversations' in converted:
        print("\nHuman message preview:")
        print(converted['conversations'][0]['value'][:200] + "...")
        print("\nAssistant message:")
        print(converted['conversations'][1]['value'])
        
        # Check if image was handled
        print("\nImage field added:", 'image' in converted)
        if 'image' in converted:
            print("Image path:", converted['image'])
        
        # Check if <image> tag was inserted
        has_image_tag = '<image>' in converted['conversations'][0]['value']
        print("Has <image> tag in question:", has_image_tag)
    
    print("\n" + "="*50)
    print("Testing iterator with actual data...")
    
    # Test the iterator
    data_iter = iter(dataset)
    try:
        sample = next(data_iter)
        print("Successfully loaded sample from iterator")
        print("Sample keys:", list(sample.keys()))
        print("Number of text_ids_list:", len(sample.get('text_ids_list', [])))
        print("Number of image_tensor_list:", len(sample.get('image_tensor_list', [])))
        print("Sequence plan length:", len(sample.get('sequence_plan', [])))
        
        # Print sequence plan details
        print("\nSequence plan:")
        for i, plan in enumerate(sample.get('sequence_plan', [])):
            print(f"  {i}: type={plan['type']}, loss={plan['loss']}")
            
    except Exception as e:
        print(f"Error loading sample: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_thinktrace_conversion()