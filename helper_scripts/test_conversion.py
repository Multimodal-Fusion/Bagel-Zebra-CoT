#!/usr/bin/env python3
"""Test the convert_thinktrace_to_conversations function directly"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load a sample from the actual data
with open('/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/frozenlake/test/frozenlake_thinktrace.jsonl', 'r') as f:
    sample_line = f.readline()
    sample_data = json.loads(sample_line)

print("Original data keys:", list(sample_data.keys()))
print("\nSample Question:", sample_data.get('Question', '')[:100] + "...")
print("Has problem_image_1:", 'problem_image_1' in sample_data)
print("problem_image_1 value:", sample_data.get('problem_image_1', 'NOT FOUND'))

# Simulate the conversion function
def convert_thinktrace_to_conversations(data_item, trace_field='Text Reasoning Trace[textual-cot]'):
    """Convert thinktrace format to conversation format"""
    import re
    
    # Extract fields
    prompt = "You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. Think step by step and generate visual aids to enhance your problem-solving. You should first think about the reasoning and planning process in the mind before generating visual aids. Wrap your text reasoning with <think></think> tokens, and wrap your final conclusion with <answer></answer> tokens. Provide your final conclusion clearly in the format of '<answer>Final Answer: <answer here></answer>'"
    
    question = data_item.get('Question', '')
    reasoning_trace = data_item.get(trace_field, '')
    final_answer = data_item.get('Final Answer', '')
    
    print(f"\nExtracted fields:")
    print(f"  Question length: {len(question)}")
    print(f"  Reasoning trace from '{trace_field}': {len(reasoning_trace)} chars")
    print(f"  Final answer: '{final_answer}'")
    
    # Format the assistant response with template
    assistant_response = ""
    if reasoning_trace:
        assistant_response = f"<think>{reasoning_trace}</think>\n"
    assistant_response += f"The final answer is {final_answer}\n"
    assistant_response += f"<ANSWER>{final_answer}</ANSWER>"
    
    # Handle image references in question
    question_with_image = question
    if 'problem_image_1' in data_item:
        print("  Found problem_image_1, replacing image reference...")
        question_with_image = re.sub(r'<image_start>\[problem_image_1\]<image_end>', '<image>', question)
        if '<image>' in question_with_image:
            print("  Successfully replaced with <image> tag")
        else:
            print("  WARNING: No image reference found to replace!")
    
    # Create clean conversation format
    converted = {
        'conversations': [
            {'from': 'human', 'value': prompt + '\n' + 'Question: ' + question_with_image},
            {'from': 'gpt', 'value': assistant_response}
        ]
    }
    
    # Add image field if present (as single string, not list)
    if 'problem_image_1' in data_item:
        converted['image'] = data_item['problem_image_1']
    
    # Add id if present, otherwise generate one
    if 'id' in data_item:
        converted['id'] = data_item['id']
    else:
        # Simple id generation based on question content
        converted['id'] = abs(hash(question)) % 1000000
    
    return converted

# Test with different trace fields
print("\n" + "="*60)
print("Testing with Text Reasoning Trace[sft]:")
converted_sft = convert_thinktrace_to_conversations(sample_data, 'Text Reasoning Trace[sft]')
print("\nConverted keys:", list(converted_sft.keys()))
print("Has conversations:", 'conversations' in converted_sft)
print("Has image:", 'image' in converted_sft)
if 'image' in converted_sft:
    print("Image value:", converted_sft['image'])
print("Has id:", 'id' in converted_sft)

print("\n" + "="*60)
print("Testing with Text Reasoning Trace[textual-cot]:")
converted_cot = convert_thinktrace_to_conversations(sample_data, 'Text Reasoning Trace[textual-cot]')
print("\nConverted keys:", list(converted_cot.keys()))
print("Assistant response preview:", converted_cot['conversations'][1]['value'][:200] + "...")

# Check if the format matches expected
print("\n" + "="*60)
print("Format validation:")
print("Conversations is list:", isinstance(converted_cot.get('conversations'), list))
print("Has 2 conversations:", len(converted_cot.get('conversations', [])) == 2)
if len(converted_cot.get('conversations', [])) == 2:
    print("First is human:", converted_cot['conversations'][0].get('from') == 'human')
    print("Second is gpt:", converted_cot['conversations'][1].get('from') == 'gpt')
    print("Human has value:", 'value' in converted_cot['conversations'][0])
    print("GPT has value:", 'value' in converted_cot['conversations'][1])
    print("Has <image> tag in human:", '<image>' in converted_cot['conversations'][0]['value'])

print("\n" + "="*60)
print("Output as JSON (truncated):")
output = json.dumps(converted_cot, indent=2)
print(output[:1000] + "..." if len(output) > 1000 else output)