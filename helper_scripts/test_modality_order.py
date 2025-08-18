#!/usr/bin/env python3
"""Test script to verify the new modality order preservation in think_trace_dataset.py"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset
from transformers import AutoTokenizer
from data.data_utils import add_special_tokens

def test_modality_ordering():
    """Test that the new parse_modality_sequence preserves original order"""
    
    # Load a sample row
    filename = './visual_data/data/frozenlake_4/test/frozenlake_4_thinktrace.jsonl'
    
    print("="*60)
    print("TESTING MODALITY ORDER PRESERVATION")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        print("Please ensure you're running from the correct directory")
        return
    
    with open(filename, 'r') as f:
        sample_row = f.readline()
        data_item = json.loads(sample_row)
    
    # Print original structure
    print("\n1. ORIGINAL DATA:")
    print("-"*40)
    print(f"Question: {data_item.get('Question', '')[:100]}...")
    print(f"Final Answer: {data_item.get('Final Answer', '')}")
    
    trace = data_item.get('Text Reasoning Trace[visual-cot]', '')
    print(f"\nOriginal Trace Length: {len(trace)} chars")
    print(f"First 200 chars: {trace[:200]}...")
    
    # Count images in original trace
    import re
    image_pattern = r'<image_start>\[([^\]]+)\]<image_end>'
    original_images = re.findall(image_pattern, trace)
    print(f"\nNumber of images in trace: {len(original_images)}")
    if original_images:
        print(f"Image references: {original_images}")
    
    # Initialize tokenizer with special tokens
    print("\n2. INITIALIZING TOKENIZER:")
    print("-"*40)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B')
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    print(f"Added {num_new_tokens} special tokens")
    print(f"Special token IDs: {new_token_ids}")
    
    # Create mock transforms that return dummy tensors
    import torch
    class MockTransform:
        def __init__(self, stride=16):
            self.stride = stride
        def __call__(self, image):
            # Return a dummy tensor with shape [C, H, W]
            return torch.zeros(3, 256, 256)
    
    # Create a test dataset instance
    dataset = ThinkTraceJSONLIterableDataset(
        dataset_name='test',
        transform=MockTransform(stride=16),  # Mock VAE transform
        tokenizer=tokenizer,
        vit_transform=MockTransform(stride=16),  # Mock VIT transform
        jsonl_path_list=[filename],
        data_dir_list=['./visual_data/data/frozenlake_4/test/'],
        num_used_data=[1],
        image_prefix_dir='./visual_data/',
        trace_field='Text Reasoning Trace[visual-cot]'
    )
    
    # Test the new parse_modality_sequence method
    print("\n3. TESTING parse_modality_sequence:")
    print("-"*40)
    segments = dataset.parse_modality_sequence(trace)
    
    print(f"Total segments: {len(segments)}")
    print("\nSegment order:")
    for i, (modality, content) in enumerate(segments):
        if modality == 'text':
            # Clean up for display
            clean_content = dataset.remove_thought_patterns(content)
            preview = clean_content[:80].replace('\n', ' ') + '...' if len(clean_content) > 80 else clean_content.replace('\n', ' ')
            print(f"  {i:2d}. TEXT: {preview}")
        else:
            print(f"  {i:2d}. IMAGE: {content}")
    
    # Verify ordering is preserved
    print("\n4. VERIFICATION:")
    print("-"*40)
    
    # Check if text-image pattern matches original
    text_count = sum(1 for m, _ in segments if m == 'text')
    image_count = sum(1 for m, _ in segments if m == 'image')
    print(f"Text segments: {text_count}")
    print(f"Image segments: {image_count}")
    
    # Test the full parse_row
    print("\n5. TESTING FULL parse_row:")
    print("-"*40)
    try:
        parsed_data = dataset.parse_row(sample_row)
        
        if parsed_data and 'sequence_plan' in parsed_data:
            print(f"Successfully parsed! Sequence has {len(parsed_data['sequence_plan'])} items")
            
            # Check sequence structure
            for i, item in enumerate(parsed_data['sequence_plan']):
                type_str = f"{item['type']:12s}"
                loss_str = f"loss={item['loss']}"
                cfg_str = f"cfg={item['enable_cfg']}"
                
                special_info = ""
                if item.get('special_token_loss') == 1:
                    special_info = f" -> predicts {item.get('special_token_label')}"
                
                print(f"  {i:2d}. {type_str} {loss_str} {cfg_str}{special_info}")
            
            # Check final answer placement
            print("\n6. FINAL ANSWER PLACEMENT:")
            print("-"*40)
            
            # Find last text segment
            text_indices = [i for i, item in enumerate(parsed_data['sequence_plan']) 
                          if item['type'] == 'text' and item['loss'] == 1]
            
            if text_indices:
                last_text_idx = text_indices[-1]
                # Count text segments up to this point
                text_count = sum(1 for item in parsed_data['sequence_plan'][:last_text_idx+1] 
                               if item['type'] == 'text')
                
                last_text_ids = parsed_data['text_ids_list'][text_count-1]
                last_text = tokenizer.decode(last_text_ids)
                
                print(f"Last text segment (index {last_text_idx}):")
                print(f"Length: {len(last_text)} chars")
                
                # Show last 400 chars to see the structure
                display_text = last_text[-400:] if len(last_text) > 400 else last_text
                print(f"\nLast part of text:\n{display_text}")
                
                # Verify structure
                has_think_close = "</think>" in last_text
                has_final_answer = "The final answer is" in last_text
                has_answer_tag = "<ANSWER>" in last_text
                
                print("\nValidation:")
                print(f"  ✓ Has </think> tag: {has_think_close}")
                print(f"  ✓ Has final answer text: {has_final_answer}")
                print(f"  ✓ Has <ANSWER> tag: {has_answer_tag}")
                
                if has_think_close and has_final_answer:
                    # Check order
                    think_pos = last_text.rfind("</think>")
                    answer_pos = last_text.find("The final answer is", think_pos)
                    if answer_pos > think_pos:
                        print(f"  ✓ Final answer correctly placed after </think>")
                    else:
                        print(f"  ✗ ERROR: Final answer not after </think>")
                else:
                    print(f"  ✗ ERROR: Missing expected structure")
            else:
                print("ERROR: No text segments with loss found")
                
        else:
            print("ERROR: Failed to parse row")
            if parsed_data:
                print(f"Parsed data keys: {parsed_data.keys()}")
                
    except Exception as e:
        print(f"ERROR during parsing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_modality_ordering()