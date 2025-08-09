#!/usr/bin/env python3
"""
Zebra-CoT Dataset Visualization - Flask App
Modern, responsive web interface for browsing the dataset
"""

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
from pathlib import Path
import json
from PIL import Image
import io
import base64
import numpy as np
from functools import lru_cache
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zebra-cot-viz-2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Cache for loaded data
dataset_cache = {}

@lru_cache(maxsize=32)
def get_categories():
    """Get all dataset categories"""
    dataset_dir = Path("../datasets/Zebra-CoT")
    categories = {}
    
    for folder in dataset_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            parquet_files = list(folder.glob("*.parquet"))
            if parquet_files:
                categories[folder.name] = [str(f) for f in parquet_files]
    
    return categories

@lru_cache(maxsize=16)
def load_parquet_cached(file_path):
    """Load and cache parquet file"""
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def image_to_base64(image_data):
    """Convert image bytes to base64 for web display"""
    if image_data and isinstance(image_data, dict) and 'bytes' in image_data:
        try:
            img = Image.open(io.BytesIO(image_data['bytes']))
            # Resize if too large
            max_size = (800, 800)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    return None

def format_reasoning_text(text):
    """Format reasoning text with HTML for better display"""
    if not text:
        return ""
    
    # Split by THOUGHT patterns
    thoughts = re.split(r'(THOUGHT \d+:)', text)
    formatted = []
    
    for i, part in enumerate(thoughts):
        if part.startswith('THOUGHT'):
            formatted.append(f'<div class="thought-header">{part}</div>')
        elif part.strip():
            formatted.append(f'<div class="thought-content">{part.strip()}</div>')
    
    return ''.join(formatted)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/categories')
def api_categories():
    """Get all categories and their stats"""
    categories = get_categories()
    stats = []
    
    for cat_name, files in categories.items():
        total_samples = 0
        for file_path in files:
            df = load_parquet_cached(file_path)
            if df is not None:
                total_samples += len(df)
        
        stats.append({
            'name': cat_name,
            'files': len(files),
            'samples': total_samples,
            'icon': get_category_icon(cat_name)
        })
    
    return jsonify(sorted(stats, key=lambda x: x['name']))

@app.route('/api/category/<path:category_name>')
def api_category_files(category_name):
    """Get files in a category"""
    categories = get_categories()
    if category_name not in categories:
        return jsonify({'error': 'Category not found'}), 404
    
    files = []
    for file_path in categories[category_name]:
        df = load_parquet_cached(file_path)
        if df is not None:
            files.append({
                'path': file_path,
                'name': Path(file_path).name,
                'samples': len(df)
            })
    
    return jsonify(files)

@app.route('/api/sample')
def api_sample():
    """Get a specific sample"""
    file_path = request.args.get('file')
    index = int(request.args.get('index', 0))
    
    if not file_path:
        return jsonify({'error': 'File path required'}), 400
    
    df = load_parquet_cached(file_path)
    if df is None:
        return jsonify({'error': 'Failed to load file'}), 500
    
    if index >= len(df):
        return jsonify({'error': 'Index out of range'}), 400
    
    sample = df.iloc[index]
    
    # Prepare response
    response = {
        'index': index,
        'total': len(df),
        'question': sample.get('Question', ''),
        'reasoning': format_reasoning_text(sample.get('Text Reasoning Trace', '')),
        'answer': sample.get('Final Answer', ''),
        'problem_images': [],
        'reasoning_images': []
    }
    
    # Process problem images
    for i in range(1, 10):
        img_key = f'problem_image_{i}'
        if img_key in sample and sample[img_key] is not None:
            img_b64 = image_to_base64(sample[img_key])
            if img_b64:
                response['problem_images'].append({
                    'id': i,
                    'data': img_b64
                })
    
    # Process reasoning images
    for i in range(1, 10):
        img_key = f'reasoning_image_{i}'
        if img_key in sample and sample[img_key] is not None:
            img_b64 = image_to_base64(sample[img_key])
            if img_b64:
                response['reasoning_images'].append({
                    'id': i,
                    'data': img_b64
                })
    
    # Add metadata
    response['metadata'] = {
        'question_length': len(response['question']),
        'reasoning_length': len(sample.get('Text Reasoning Trace', '')),
        'answer_length': len(response['answer']),
        'problem_image_count': len(response['problem_images']),
        'reasoning_image_count': len(response['reasoning_images'])
    }
    
    return jsonify(response)

@app.route('/api/random')
def api_random():
    """Get a random sample from a category"""
    category = request.args.get('category')
    
    if not category:
        # Random from any category
        categories = get_categories()
        category = np.random.choice(list(categories.keys()))
    
    categories = get_categories()
    if category not in categories:
        return jsonify({'error': 'Category not found'}), 404
    
    # Random file from category
    file_path = np.random.choice(categories[category])
    df = load_parquet_cached(file_path)
    
    if df is None:
        return jsonify({'error': 'Failed to load file'}), 500
    
    # Random index
    index = np.random.randint(0, len(df))
    
    return jsonify({
        'category': category,
        'file': file_path,
        'index': index
    })

@app.route('/api/stats')
def api_stats():
    """Get overall dataset statistics"""
    categories = get_categories()
    total_samples = 0
    total_files = 0
    category_breakdown = []
    
    for cat_name, files in categories.items():
        cat_samples = 0
        for file_path in files:
            df = load_parquet_cached(file_path)
            if df is not None:
                cat_samples += len(df)
        
        total_samples += cat_samples
        total_files += len(files)
        
        category_breakdown.append({
            'name': cat_name,
            'samples': cat_samples,
            'percentage': 0  # Will calculate after
        })
    
    # Calculate percentages
    for cat in category_breakdown:
        cat['percentage'] = round((cat['samples'] / total_samples) * 100, 1) if total_samples > 0 else 0
    
    return jsonify({
        'total_categories': len(categories),
        'total_files': total_files,
        'total_samples': total_samples,
        'category_breakdown': sorted(category_breakdown, key=lambda x: x['samples'], reverse=True)
    })

def get_category_icon(category_name):
    """Get an appropriate icon for each category"""
    icons = {
        'Physics': '⚛️',
        'Chemistry': '🧪',
        'Geometry': '📐',
        'Chess': '♟️',
        'Checkers': '⚫',
        'Tetris': '🎮',
        'Maze': '🌀',
        'Robot': '🤖',
        'Visual': '👁️',
        'Graph': '📊',
        'Competitive': '💻',
        'Connect': '🔴',
        'Ciphers': '🔐',
        'RPM': '🧩',
        'ARC': '🎯',
        'Search': '🔍',
        'Jigsaw': '🧩',
        'Counting': '🔢',
        'Embodied': '🚶'
    }
    
    for key, icon in icons.items():
        if key in category_name:
            return icon
    return '📁'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5020)
    args = parser.parse_args()  
    app.run(debug=True, port=args.port, host='0.0.0.0')