#!/usr/bin/env python3
"""
General evaluation script for thinktrace datasets
"""

import os
import json
import argparse
import shutil
from datetime import datetime
from copy import deepcopy
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

def load_dataset(jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load dataset from JSONL file"""
    samples = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples

def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model output"""
    if '<ANSWER>' in text and '</ANSWER>' in text:
        start = text.find('<ANSWER>') + 8
        end = text.find('</ANSWER>')
        return text[start:end].strip()
    elif '<answer>' in text and '</answer>' in text:
        start = text.find('<answer>') + 8
        end = text.find('</answer>')
        return text[start:end].strip()
    elif 'Final Answer:' in text:
        return text.split('Final Answer:')[1].strip().split('\n')[0].strip()
    return None

def setup_model(checkpoint_dir: str, args):
    """Initialize model, tokenizer, and related components"""
    
    # Check checkpoint
    checkpoint_file = "ema.safetensors"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(checkpoint_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(checkpoint_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(checkpoint_dir, "ae.safetensors"))
    
    # Bagel config
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )
    
    # Create model with empty weights
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # Image transforms
    dim=args.image_dim
    vae_transform = ImageTransform(dim, dim, 16)
    vit_transform = ImageTransform(dim, dim, 14)
    
    # Device mapping
    max_mem_per_gpu = "80GiB"
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        dtype=torch.bfloat16,
    )
    
    # Handle same-device modules
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    
    if torch.cuda.device_count() == 1:
        for k in same_device_modules:
            device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        if first_device is not None:
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
    
    # Load checkpoint
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
    )
    
    model = model.eval()
    print('Model loaded successfully!')
    
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids

def run_inference(
    inferencer: InterleaveInferencer,
    question: str,
    problem_image: Optional[Image.Image],
    visual_cot: bool,
    max_inference_steps: int,
    inference_hyper: Dict[str, Any]
) -> Dict[str, Any]:
    """Run inference on a single sample"""
    
    SYSTEM_PROMPT = '''You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. Think step by step and use visual aids to enhance your problem-solving. Provide your final conclusion clearly in the format of "Final Answer: <answer here>"'''
    
    # Extract question without image tags
    prompt = question.split('<image_start>')[0].strip()
    
    # Initialize input
    if problem_image is not None:
        current_input = [prompt, problem_image]
    else:
        current_input = [prompt]
    
    reasoning_text = []
    generated_image = None
    
    # Step 1: Visual CoT generation (if enabled)
    if visual_cot and max_inference_steps > 0:
        # Generate image first
        output = inferencer.interleave_inference(
            current_input,
            system_prompt=SYSTEM_PROMPT,
            **inference_hyper
        )
        if output and isinstance(output[0], Image.Image):
            generated_image = output[0]
            current_input = current_input + [generated_image]
    
    # Step 2: Text generation
    if max_inference_steps > 0:
        output = inferencer.interleave_inference(
            current_input,
            understanding_output=True,
            system_prompt=SYSTEM_PROMPT,
            **inference_hyper
        )
        
        if output and output[0]:
            # Extract text from response
            text = output[0]
            if '<|im_start|>' in text and '<|im_end|>' in text:
                extracted_text = text.split('<|im_end|>')[0].split('<|im_start|>')[1]
            else:
                extracted_text = text
            reasoning_text.append(extracted_text)
    
    # Extract final answer
    final_answer = None
    for text in reasoning_text:
        answer = extract_answer(text)
        if answer:
            final_answer = answer
            break
    
    return {
        'reasoning_text': reasoning_text,
        'generated_image': generated_image,
        'final_answer': final_answer
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on thinktrace dataset")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the thinktrace JSONL file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./visual_data",
        help="Base directory for dataset images (default: ./visual_data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    parser.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Evaluation run name for output subfolder"
    )
    parser.add_argument(
        "--visual-cot",
        action=argparse.BooleanOptionalAction, # also support --no-visual-cot
        default=False,
        help="Enable visual CoT generation"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--max-inference-steps",
        type=int,
        default=1,
        help="Maximum inference steps (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing output directory if it exists"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--image-dim",
        type=int,
        default=512,
        help="Image dimension (default: 512)"
    )
    
    
    args = parser.parse_args()
    
    # Generate eval name if not provided
    if args.eval_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.basename(args.dataset_path).replace('.jsonl', '')
        args.eval_name = f"eval_{dataset_name}_{timestamp}"
    
    # Setup output directory
    output_path = os.path.join(args.output_dir, args.eval_name)
    if os.path.exists(output_path):
        if args.replace:
            shutil.rmtree(output_path)
            os.makedirs(output_path)
        else:
            print(f"Output directory {output_path} already exists. Use --replace to overwrite.")
            return
    else:
        os.makedirs(output_path, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    # Set random seed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_path}")
    samples = load_dataset(args.dataset_path, args.max_samples)
    print(f"Loaded {len(samples)} samples")
    
    # Setup model
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = setup_model(
        args.checkpoint_dir, args
    )
    
    # Initialize inferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )
    
    # Inference hyperparameters
    inference_hyper = dict(
        do_sample=True,
        text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=25,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )
    
    # Run evaluation
    results = []
    correct = 0
    total = 0
    
    print("\n" + "="*60)
    print("Starting Evaluation")
    print("="*60)
    
    # create a folder to save generated images if visual-cot is enabled
    if args.visual_cot:
        generated_image_folder = os.path.join(output_path, f"generated_images")
        os.makedirs(generated_image_folder, exist_ok=True)

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        # Extract sample data
        question = sample.get("Question", "")
        expected_answer = sample.get("Final Answer", "N/A")
        
        # Load problem image if exists
        problem_image = None
        problem_image_path = None
        if "problem_image_1" in sample:
            problem_image_path = os.path.join(args.data_dir, sample["problem_image_1"])
            if os.path.exists(problem_image_path):
                problem_image = Image.open(problem_image_path)
        
        # Run inference
        inference_result = run_inference(
            inferencer=inferencer,
            question=question,
            problem_image=problem_image,
            visual_cot=args.visual_cot,
            max_inference_steps=args.max_inference_steps,
            inference_hyper=inference_hyper
        )
        
        # Check correctness
        is_correct = inference_result['final_answer'] == expected_answer
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        result = {
            "sample_idx": idx,
            "question": question[:200] + "..." if len(question) > 200 else question,
            "problem_image": sample.get("problem_image_1"),
            "expected_answer": expected_answer,
            "predicted_answer": inference_result['final_answer'],
            "correct": is_correct,
            "reasoning_text": inference_result['reasoning_text'],
            "has_generated_image": inference_result['generated_image'] is not None
        }
        
        # save generated images for each sample in a folder
        if args.visual_cot and inference_result['generated_image']:
            sample_image_path = os.path.join(generated_image_folder, f"sample_{idx}_generated_0.png")
            inference_result['generated_image'].save(sample_image_path)
            result['saved_generated_image'] = sample_image_path
        
        results.append(result)
        
        # Print progress every 10 samples
        if (idx + 1) % 10 == 0:
            current_acc = correct / total * 100
            print(f"Progress: {idx+1}/{len(samples)}, Accuracy: {current_acc:.2f}%")
    
    # Calculate final metrics
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Save results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_dir": args.checkpoint_dir,
            "dataset_path": args.dataset_path,
            "data_dir": args.data_dir,
            "total_samples": len(samples),
            "evaluated_samples": total,
            "visual_cot": args.visual_cot,
            "max_inference_steps": args.max_inference_steps,
            "seed": args.seed
        },
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        },
        "results": results
    }
    
    # Save to JSON
    results_path = os.path.join(output_path, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {results_path}")
    print("="*60)

if __name__ == "__main__":
    main()