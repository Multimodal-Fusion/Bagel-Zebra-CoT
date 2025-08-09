#!/usr/bin/env python3
"""
Simple test script for ThinkTrace dataset without full dataloader.
This demonstrates the key concepts without all the dependencies.
"""

import json
from pathlib import Path
import sys

# Add Bagel to path for tokenizer
bagel_root = Path("/home/colligo/project/vlm/Bagel")
sys.path.insert(0, str(bagel_root))

from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens


def visualize_thinktrace_jsonl():
    """Visualize the ThinkTrace JSONL format and demonstrate the concept."""
    
    jsonl_path = Path("./datasets/chess_thinktrace.jsonl")
    
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found!")
        print("Run 'python convert_zebra_to_thinktrace.py' first to create it")
        return
    
    print("\n" + "="*80)
    print("THINKTRACE DATASET VISUALIZATION")
    print("="*80)
    print("\nThinkTrace is a special dataset format for training models that can:")
    print("1. Think step-by-step with chain-of-thought reasoning")
    print("2. Generate visual aids during reasoning")
    print("3. Handle interleaved text and images")
    print("="*80)
    
    # Load tokenizer
    tokenizer_path = "/home/colligo/project/vlm/Bagel/hf/Qwen2.5-0.5B-Instruct/"
    try:
        tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
        tokenizer = add_special_tokens(tokenizer)
        print(f"\n✅ Tokenizer loaded from {tokenizer_path}")
    except Exception as e:
        print(f"\n⚠️  Could not load tokenizer: {e}")
        tokenizer = None
    
    # Read JSONL samples
    with open(jsonl_path, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(samples)}")
    
    # Analyze first sample
    if samples:
        sample = samples[0]
        
        print(f"\n📝 FIRST SAMPLE ANALYSIS:")
        print(f"  Fields: {list(sample.keys())}")
        
        # Extract image references
        question = sample['Question']
        trace = sample['Text Reasoning Trace']
        answer = sample['Final Answer']
        
        # Count image references
        question_images = question.count('<image_start>')
        trace_images = trace.count('<image_start>')
        
        print(f"\n  Image Usage:")
        print(f"    Images in question: {question_images}")
        print(f"    Images in reasoning: {trace_images}")
        
        # Count thought steps
        thought_count = trace.count('THOUGHT')
        
        print(f"\n  Reasoning Structure:")
        print(f"    Number of thought steps: {thought_count}")
        
        # Show question
        print(f"\n  📋 Question:")
        clean_question = question.replace('<image_start>', '[IMG:').replace('<image_end>', ']')
        print(f"    {clean_question[:200]}...")
        
        # Show reasoning pattern
        print(f"\n  🤔 Reasoning Pattern:")
        thoughts = trace.split('THOUGHT')[:3]  # First 2 thoughts
        for i, thought in enumerate(thoughts[1:], 1):
            clean_thought = thought.replace('<image_start>', '[IMG:').replace('<image_end>', ']')
            preview = clean_thought[:150].replace('\n', ' ')
            print(f"    THOUGHT {i}: {preview}...")
        
        # Show answer
        print(f"\n  ✅ Final Answer: {answer[:100]}")
        
        # Show how this would be processed
        print(f"\n🔄 PROCESSING FLOW:")
        print("  1. Question with images → VIT encoder (understanding)")
        print("  2. Text reasoning → Language model generates <think> tokens")
        print("  3. Visual reasoning → VAE decoder generates images")
        print("  4. Final answer → Language model generates <answer> tokens")
        
        print(f"\n🎯 TRAINING OBJECTIVES:")
        print("  • CE Loss: Applied to text generation (reasoning + answer)")
        print("  • MSE Loss: Applied to image generation during reasoning")
        print("  • Model learns to coordinate visual and textual thinking")
        
        # Demonstrate sequence structure
        print(f"\n📦 SEQUENCE STRUCTURE (conceptual):")
        print("  [BOS] [Question + Images] [<think>] [Reasoning + Generated Images] [</think>]")
        print("        [<answer>] [Final Answer] [</answer>] [EOS]")
        print("\n  Where:")
        print("    • Question images use VIT for understanding")
        print("    • Generated images use VAE for creation")
        print("    • Text uses language model with CE loss")
        print("    • Images use diffusion with MSE loss")


def main():
    print("\n" + "="*80)
    print("THINKTRACE DATASET - SIMPLE VISUALIZATION")
    print("="*80)
    
    # Check if data exists
    jsonl_path = Path("./datasets/chess_thinktrace.jsonl")
    if not jsonl_path.exists():
        print("\n⚠️  Dataset not found. Running conversion first...")
        import subprocess
        result = subprocess.run([sys.executable, "convert_zebra_to_thinktrace.py"])
        if result.returncode != 0:
            print("Conversion failed!")
            return
    
    # Visualize
    visualize_thinktrace_jsonl()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n🌟 What makes ThinkTrace special:")
    print("1. Interleaved text and image generation during reasoning")
    print("2. Explicit thinking tokens (<think></think>) for chain-of-thought")
    print("3. Dual loss training (CE for text, MSE for images)")
    print("4. Model learns to 'think visually' by generating diagrams")
    print("\n💡 This enables models to:")
    print("• Solve complex visual reasoning tasks")
    print("• Generate explanatory diagrams")
    print("• Provide step-by-step visual proofs")
    print("• Bridge visual and textual understanding")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()