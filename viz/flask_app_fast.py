#!/usr/bin/env python3
"""
Zebra-CoT Dataset Visualization - Optimized Flask App
Fast loading with lazy data access
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
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zebra-cot-viz-2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Global cache for current file only
current_df_cache = {'file': None, 'df': None}

def get_categories_fast():
    """Get categories without loading any data - just file counts"""
    dataset_dir = Path("../datasets/Zebra-CoT")
    categories = {}
    
    for folder in dataset_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            parquet_files = list(folder.glob("*.parquet"))
            if parquet_files:
                # Don't load files, just count them
                categories[folder.name] = {
                    'files': [str(f) for f in parquet_files],
                    'file_count': len(parquet_files),
                    'estimated_samples': len(parquet_files) * 3000  # Rough estimate
                }
    
    return categories

def load_single_file(file_path):
    """Load a single parquet file with caching"""
    global current_df_cache
    
    # Check cache
    if current_df_cache['file'] == file_path:
        return current_df_cache['df']
    
    try:
        print(f"Loading file: {file_path}")
        df = pd.read_parquet(file_path)
        # Update cache
        current_df_cache['file'] = file_path
        current_df_cache['df'] = df
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_file_info(file_path):
    """Get basic file info without loading the full dataframe"""
    try:
        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        # Try to get row count without loading full data
        # This is a quick hack - just return estimate based on file size
        estimated_rows = int(file_size * 30)  # Rough estimate
        
        return {
            'path': str(file_path),
            'name': Path(file_path).name,
            'size_mb': round(file_size, 2),
            'estimated_samples': estimated_rows
        }
    except:
        return {
            'path': str(file_path),
            'name': Path(file_path).name,
            'size_mb': 0,
            'estimated_samples': 0
        }

def image_to_base64_fast(image_data, max_size=(600, 600)):
    """Convert image bytes to base64 with aggressive compression"""
    if image_data and isinstance(image_data, dict) and 'bytes' in image_data:
        try:
            img = Image.open(io.BytesIO(image_data['bytes']))
            
            # Aggressive resize for faster loading
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            
            # Lower quality for faster loading
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    return None

def format_reasoning_text(text, max_length=5000):
    """Format and truncate reasoning text"""
    if not text:
        return ""
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"
    
    # Simple formatting
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
    return render_template('index_fast.html')

@app.route('/api/categories')
def api_categories():
    """Get categories without loading data"""
    categories = get_categories_fast()
    stats = []
    
    for cat_name, cat_info in categories.items():
        stats.append({
            'name': cat_name,
            'files': cat_info['file_count'],
            'samples': cat_info['estimated_samples'],
            'icon': get_category_icon(cat_name)
        })
    
    return jsonify(sorted(stats, key=lambda x: x['name']))

@app.route('/api/category/<path:category_name>/files')
def api_category_files(category_name):
    """Get file list for a category without loading data"""
    categories = get_categories_fast()
    if category_name not in categories:
        return jsonify({'error': 'Category not found'}), 404
    
    files = []
    for file_path in categories[category_name]['files']:
        files.append(get_file_info(file_path))
    
    return jsonify(files)

@app.route('/api/file/info')
def api_file_info():
    """Get file info and sample count"""
    file_path = request.args.get('file')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400
    
    df = load_single_file(file_path)
    if df is None:
        return jsonify({'error': 'Failed to load file'}), 500
    
    return jsonify({
        'path': file_path,
        'samples': len(df),
        'columns': df.columns.tolist()
    })

@app.route('/api/sample')
def api_sample():
    """Get a specific sample - optimized version"""
    file_path = request.args.get('file')
    index = int(request.args.get('index', 0))
    
    if not file_path:
        return jsonify({'error': 'File path required'}), 400
    
    df = load_single_file(file_path)
    if df is None:
        return jsonify({'error': 'Failed to load file'}), 500
    
    if index >= len(df):
        return jsonify({'error': 'Index out of range'}), 400
    
    sample = df.iloc[index]
    
    # Prepare response with smaller images
    response = {
        'index': index,
        'total': len(df),
        'question': sample.get('Question', '')[:2000],  # Limit text length
        'reasoning': format_reasoning_text(sample.get('Text Reasoning Trace', '')),
        'answer': sample.get('Final Answer', '')[:2000],
        'problem_images': [],
        'reasoning_images': []
    }
    
    # Only load first 4 images for faster loading
    for i in range(1, 5):
        img_key = f'problem_image_{i}'
        if img_key in sample and sample[img_key] is not None:
            img_b64 = image_to_base64_fast(sample[img_key])
            if img_b64:
                response['problem_images'].append({'id': i, 'data': img_b64})
    
    for i in range(1, 5):
        img_key = f'reasoning_image_{i}'
        if img_key in sample and sample[img_key] is not None:
            img_b64 = image_to_base64_fast(sample[img_key])
            if img_b64:
                response['reasoning_images'].append({'id': i, 'data': img_b64})
    
    # Add metadata
    response['metadata'] = {
        'question_length': len(sample.get('Question', '')),
        'reasoning_length': len(sample.get('Text Reasoning Trace', '')),
        'answer_length': len(sample.get('Final Answer', '')),
        'problem_image_count': len(response['problem_images']),
        'reasoning_image_count': len(response['reasoning_images'])
    }
    
    return jsonify(response)

@app.route('/api/sample/quick')
def api_sample_quick():
    """Get sample without images for quick preview"""
    file_path = request.args.get('file')
    index = int(request.args.get('index', 0))
    
    if not file_path:
        return jsonify({'error': 'File path required'}), 400
    
    df = load_single_file(file_path)
    if df is None:
        return jsonify({'error': 'Failed to load file'}), 500
    
    if index >= len(df):
        return jsonify({'error': 'Index out of range'}), 400
    
    sample = df.iloc[index]
    
    # Text only response
    response = {
        'index': index,
        'total': len(df),
        'question': sample.get('Question', '')[:500],
        'answer': sample.get('Final Answer', '')[:500],
        'has_images': any(f'problem_image_{i}' in sample and sample[f'problem_image_{i}'] is not None for i in range(1, 10))
    }
    
    return jsonify(response)

@app.route('/api/random')
def api_random():
    """Get a random sample"""
    category = request.args.get('category')
    
    categories = get_categories_fast()
    
    if not category:
        category = np.random.choice(list(categories.keys()))
    
    if category not in categories:
        return jsonify({'error': 'Category not found'}), 404
    
    # Random file
    file_path = np.random.choice(categories[category]['files'])
    
    # Load file to get actual count
    df = load_single_file(file_path)
    if df is None:
        return jsonify({'error': 'Failed to load file'}), 500
    
    index = np.random.randint(0, len(df))
    
    return jsonify({
        'category': category,
        'file': file_path,
        'index': index
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
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(debug=True, port=args.port, host='0.0.0.0')