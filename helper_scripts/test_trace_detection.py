#!/usr/bin/env python3
"""Test script to verify trace detection for FrozenLake dataset."""

import json

# Test sample structure
sample = {
    "Question": "Test question",
    "Text Reasoning Trace[sft]": "",  # Empty SFT
    "Text Reasoning Trace[textual-cot]": "Some textual reasoning",
    "Text Reasoning Trace[visual-cot]": "Some visual reasoning with images",
    "Final Answer": "A"
}

def get_available_traces(sample):
    """Get all available reasoning trace types in a sample."""
    trace_types = []
    
    # Check for different trace formats - include even if empty
    if 'Text Reasoning Trace' in sample:
        trace_types.append(('original', 'Original'))
    
    if 'Text Reasoning Trace[sft]' in sample:
        trace_types.append(('sft', 'SFT'))
    
    if 'Text Reasoning Trace[textual-cot]' in sample:
        trace_types.append(('textual-cot', 'Textual CoT'))
    
    if 'Text Reasoning Trace[visual-cot]' in sample:
        trace_types.append(('visual-cot', 'Visual CoT'))
    
    return trace_types

# Test the function
traces = get_available_traces(sample)
print("Detected traces:")
for trace_type, label in traces:
    value = sample.get(f'Text Reasoning Trace[{trace_type}]' if trace_type != 'original' else 'Text Reasoning Trace', '')
    is_empty = not value or value.strip() == ''
    print(f"  - {label} ({trace_type}): {'EMPTY' if is_empty else 'Has content'}")

print(f"\nTotal traces found: {len(traces)}")
print("Expected: 3 (SFT, Textual CoT, Visual CoT)")