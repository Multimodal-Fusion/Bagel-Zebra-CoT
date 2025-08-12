#!/usr/bin/env python3
"""
Flask visualization app for ThinkTrace format JSONL files.
Usage: python thinktrace_viz_app.py --jsonl <jsonl_file> --data-dir <data_directory>
"""

import argparse
import json
import os
import base64
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Global variables to store data
samples = []
data_dir = None
jsonl_file = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThinkTrace Visualizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .sample-card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .navigation-controls {
            position: sticky;
            top: 10px;
            z-index: 100;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .chess-image {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            margin: 10px 0;
        }
        .thought-section {
            background: #f1f3f5;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        .answer-options {
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .answer-option {
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        .final-answer {
            background: #d4edda;
            border: 2px solid #28a745;
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.2em;
            text-align: center;
            margin: 20px 0;
        }
        .reasoning-images {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .reasoning-image-container {
            text-align: center;
        }
        .reasoning-image-container img {
            width: 100%;
            max-width: 400px;
        }
        .btn-nav {
            min-width: 100px;
        }
        .sample-info {
            background: #e7f3ff;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .error-message {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
        }
        .loading {
            text-align: center;
            padding: 50px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .image-label {
            font-weight: bold;
            color: #495057;
            margin-top: 15px;
            margin-bottom: 5px;
        }
        .collapsed-text {
            max-height: 200px;
            overflow: hidden;
            position: relative;
        }
        .collapsed-text.expanded {
            max-height: none;
        }
        .expand-btn {
            cursor: pointer;
            color: #667eea;
            text-decoration: underline;
            margin-top: 10px;
        }
        .plain-format {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            font-family: monospace;
            white-space: pre-wrap;
            line-height: 1.6;
        }
        .plain-image-marker {
            color: #dc3545;
            font-weight: bold;
            background: #ffe0e0;
            padding: 2px 6px;
            border-radius: 3px;
            margin: 5px 0;
            display: inline-block;
        }
        .view-toggle {
            background: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .trace-selector {
            background: #f0f8ff;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #cce7ff;
        }
        .trace-type-badge {
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-left: 10px;
        }
        .no-trace-message {
            color: #6c757d;
            font-style: italic;
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header">
            <h1>🎯 ThinkTrace Visualizer</h1>
            <p class="mb-0">File: {{ jsonl_file }}</p>
            <p class="mb-0">Total Samples: {{ total_samples }}</p>
        </div>

        <div class="view-toggle">
            <div class="btn-group" role="group">
                <button type="button" class="btn btn-outline-primary" id="richViewBtn" onclick="setViewMode('rich')">Rich View</button>
                <button type="button" class="btn btn-outline-primary" id="plainViewBtn" onclick="setViewMode('plain')">Plain View</button>
            </div>
        </div>
        
        <div class="trace-selector" id="traceSelectorContainer" style="display: none;">
            <label for="traceTypeSelect" class="form-label">Select Reasoning Trace Type:</label>
            <select id="traceTypeSelect" class="form-select" onchange="changeTraceType()">
                <!-- Options will be populated dynamically -->
            </select>
            <small class="text-muted d-block mt-2">Available trace types for this sample</small>
        </div>

        <div class="navigation-controls">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <div class="d-flex gap-2">
                        <button id="prevBtn" class="btn btn-primary btn-nav" onclick="navigate(-1)">
                            ← Previous
                        </button>
                        <button id="nextBtn" class="btn btn-primary btn-nav" onclick="navigate(1)">
                            Next →
                        </button>
                        <input type="number" id="sampleIndex" class="form-control" style="width: 100px" 
                               min="0" max="{{ total_samples - 1 }}" value="0">
                        <button class="btn btn-success" onclick="goToSample()">Go</button>
                    </div>
                </div>
                <div class="col-md-6 text-end">
                    <span class="badge bg-secondary fs-6" id="currentIndex">Sample 1 / {{ total_samples }}</span>
                </div>
            </div>
        </div>

        <div id="sampleContent">
            <div class="loading">
                <div class="spinner"></div>
                <p class="mt-3">Loading sample...</p>
            </div>
        </div>
    </div>

    <script>
        let currentIndex = 0;
        const totalSamples = {{ total_samples }};
        let viewMode = 'rich';  // 'rich' or 'plain'
        let currentTraceType = null;
        let availableTraces = [];

        function navigate(direction) {
            currentIndex = Math.max(0, Math.min(totalSamples - 1, currentIndex + direction));
            loadSample();
        }

        function goToSample() {
            const index = parseInt(document.getElementById('sampleIndex').value);
            if (!isNaN(index) && index >= 0 && index < totalSamples) {
                currentIndex = index;
                loadSample();
            }
        }

        function setViewMode(mode) {
            viewMode = mode;
            
            // Update button states
            document.getElementById('richViewBtn').classList.toggle('active', mode === 'rich');
            document.getElementById('plainViewBtn').classList.toggle('active', mode === 'plain');
            
            // Reload current sample
            loadSample();
        }

        function loadSample() {
            document.getElementById('sampleIndex').value = currentIndex;
            document.getElementById('currentIndex').textContent = `Sample ${currentIndex + 1} / ${totalSamples}`;
            
            // Update button states
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === totalSamples - 1;

            // Show loading
            document.getElementById('sampleContent').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p class="mt-3">Loading sample ${currentIndex + 1}...</p>
                </div>
            `;

            // Fetch sample data with view mode and trace type
            let url = `/sample/${currentIndex}?view=${viewMode}`;
            if (currentTraceType) {
                url += `&trace_type=${currentTraceType}`;
            }
            
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('sampleContent').innerHTML = `
                            <div class="error-message">${data.error}</div>
                        `;
                    } else {
                        // Update available traces and selector
                        if (data.available_traces) {
                            updateTraceSelector(data.available_traces, data.current_trace_type);
                        }
                        
                        if (viewMode === 'plain') {
                            displayPlainSample(data);
                        } else {
                            displaySample(data);
                        }
                    }
                })
                .catch(error => {
                    document.getElementById('sampleContent').innerHTML = `
                        <div class="error-message">Error loading sample: ${error}</div>
                    `;
                });
        }

        function displayPlainSample(data) {
            let html = '<div class="sample-card">';
            
            // Sample info
            html += `<div class="sample-info">
                        <strong>Sample Index:</strong> ${currentIndex} | 
                        <strong>Plain Text View</strong>
                     </div>`;
            
            html += '<div class="plain-format">';
            
            // Display the raw sequence
            if (data.plain_sequence) {
                data.plain_sequence.forEach(item => {
                    if (item.type === 'text') {
                        // Escape HTML and preserve formatting
                        const escaped = item.content
                            .replace(/&/g, '&amp;')
                            .replace(/</g, '&lt;')
                            .replace(/>/g, '&gt;');
                        html += escaped;
                    } else if (item.type === 'image') {
                        html += `\n<span class="plain-image-marker">[IMAGE: ${item.name}]</span>\n`;
                        if (item.base64) {
                            html += `<img src="data:image/png;base64,${item.base64}" class="chess-image" style="max-width: 500px; margin: 10px 0;">\n`;
                        }
                    }
                });
            }
            
            html += '</div>';
            html += '</div>';
            
            document.getElementById('sampleContent').innerHTML = html;
        }

        function updateTraceSelector(traces, currentType) {
            availableTraces = traces;
            currentTraceType = currentType;
            
            const container = document.getElementById('traceSelectorContainer');
            const select = document.getElementById('traceTypeSelect');
            
            if (traces && traces.length > 0) {
                container.style.display = 'block';
                
                // Clear and populate options
                select.innerHTML = '';
                traces.forEach(trace => {
                    const option = document.createElement('option');
                    option.value = trace.type;
                    option.textContent = trace.label;
                    if (trace.type === currentType) {
                        option.selected = true;
                    }
                    select.appendChild(option);
                });
            } else {
                container.style.display = 'none';
            }
        }
        
        function changeTraceType() {
            const select = document.getElementById('traceTypeSelect');
            currentTraceType = select.value;
            loadSample();
        }
        
        function displaySample(data) {
            let html = '<div class="sample-card">';
            
            // Sample info with trace type badge
            html += `<div class="sample-info">
                        <strong>Sample Index:</strong> ${currentIndex} | 
                        <strong>Final Answer:</strong> ${data.final_answer}
                        ${data.trace_type_label ? `<span class="trace-type-badge">${data.trace_type_label}</span>` : ''}
                     </div>`;

            // Question
            html += '<h3>📝 Question</h3>';
            html += `<div class="mb-3">${formatQuestion(data.question)}</div>`;

            // Problem image
            if (data.problem_image) {
                html += '<div class="image-label">Problem Position:</div>';
                html += `<img src="data:image/png;base64,${data.problem_image}" class="chess-image">`;
            }

            // Thinking trace
            if (data.thoughts && data.thoughts.length > 0) {
                html += '<h3 class="mt-4">🤔 Thinking Process</h3>';
                html += '<div id="thoughtsContainer">';
                
                data.thoughts.forEach((thought, index) => {
                    // Determine label based on thought text
                    let label = thought.text.startsWith('Step ') ? '' : `<strong>THOUGHT ${index}:</strong><br>`;
                    
                    html += `<div class="thought-section">
                                ${label}
                                ${thought.text}
                             </div>`;
                    
                    if (thought.image) {
                        html += `<div class="reasoning-image-container mt-2">
                                    <img src="data:image/png;base64,${thought.image}" class="chess-image">
                                 </div>`;
                    }
                });
                
                html += '</div>';
            } else if (data.no_trace) {
                html += `<div class="no-trace-message">
                            ${data.no_trace_message || 'No reasoning trace available for this sample'}
                         </div>`;
            }

            // Final answer
            html += `<div class="final-answer">
                        Final Answer: ${data.final_answer}
                     </div>`;

            html += '</div>';
            
            document.getElementById('sampleContent').innerHTML = html;
        }

        function formatQuestion(question) {
            // Extract and format multiple choice options
            const lines = question.split('\\n');
            let formatted = '';
            let inOptions = false;
            
            for (let line of lines) {
                if (line.match(/^[A-D]:/)) {
                    if (!inOptions) {
                        formatted += '<div class="answer-options"><strong>Options:</strong><br>';
                        inOptions = true;
                    }
                    formatted += `<div class="answer-option">${line}</div>`;
                } else if (line.trim() !== '' && !line.includes('<image_')) {
                    if (inOptions) {
                        formatted += '</div>';
                        inOptions = false;
                    }
                    formatted += `<p>${line}</p>`;
                }
            }
            
            if (inOptions) {
                formatted += '</div>';
            }
            
            return formatted;
        }

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft' && currentIndex > 0) {
                navigate(-1);
            } else if (e.key === 'ArrowRight' && currentIndex < totalSamples - 1) {
                navigate(1);
            }
        });

        // Load first sample on page load
        window.onload = function() {
            // Set initial view mode button state
            document.getElementById('richViewBtn').classList.add('active');
            loadSample();
        };
    </script>
</body>
</html>
"""

def load_image_as_base64(image_path):
    """Load an image and convert it to base64 string."""
    try:
        full_path = Path(data_dir) / image_path
        if not full_path.exists():
            # Try without the prefix if it's already included
            full_path = Path(image_path)
        
        if full_path.exists():
            with Image.open(full_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                
                # Convert to base64
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
    
    return None

def parse_thinking_trace(trace_text):
    """Parse the thinking trace into individual thoughts with their images."""
    thoughts = []
    current_thought = None
    
    lines = trace_text.split('\n')
    
    for line in lines:
        # Handle both THOUGHT format (chess) and Step format (FrozenLake)
        if line.startswith('THOUGHT') or line.startswith('Step '):
            if current_thought:
                thoughts.append(current_thought)
            
            # Extract thought/step number and text
            if line.startswith('THOUGHT'):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_thought = {
                        'text': parts[1].strip(),
                        'image': None
                    }
            else:  # Step format
                current_thought = {
                    'text': line,
                    'image': None
                }
        elif current_thought and line.strip() and not line.startswith('<image_'):
            current_thought['text'] += '\n' + line
        elif current_thought and '<image_start>' in line:
            # Extract image reference
            import re
            match = re.search(r'\[(\w+)\]', line)
            if match:
                current_thought['image_ref'] = match.group(1)
    
    if current_thought:
        thoughts.append(current_thought)
    
    return thoughts

def get_available_traces(sample):
    """Get all available reasoning trace types in a sample."""
    trace_types = []
    
    # Check for different trace formats
    if 'Text Reasoning Trace' in sample and sample['Text Reasoning Trace']:
        trace_types.append(('original', 'Original'))
    
    if 'Text Reasoning Trace[sft]' in sample and sample['Text Reasoning Trace[sft]']:
        trace_types.append(('sft', 'SFT'))
    
    if 'Text Reasoning Trace[textual-cot]' in sample and sample['Text Reasoning Trace[textual-cot]']:
        trace_types.append(('textual-cot', 'Textual CoT'))
    
    if 'Text Reasoning Trace[visual-cot]' in sample and sample['Text Reasoning Trace[visual-cot]']:
        trace_types.append(('visual-cot', 'Visual CoT'))
    
    return trace_types

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE, 
                                 jsonl_file=os.path.basename(jsonl_file),
                                 total_samples=len(samples))

@app.route('/sample/<int:index>')
def get_sample(index):
    """Get a specific sample by index."""
    if index < 0 or index >= len(samples):
        return jsonify({'error': f'Invalid sample index: {index}'})
    
    try:
        sample = samples[index]
        view_mode = request.args.get('view', 'rich')
        requested_trace_type = request.args.get('trace_type', None)
        
        # Get available trace types
        available_traces = get_available_traces(sample)
        
        # Determine which trace to use
        trace_type = None
        trace_text = None
        trace_type_label = None
        
        if requested_trace_type:
            # Use requested trace type if available
            for key, label in available_traces:
                if key == requested_trace_type:
                    trace_type = key
                    trace_type_label = label
                    break
        
        if not trace_type and available_traces:
            # Default to first available trace
            trace_type, trace_type_label = available_traces[0]
        
        # Get the trace text based on selected type
        if trace_type:
            if trace_type == 'original':
                trace_text = sample.get('Text Reasoning Trace', '')
            elif trace_type == 'sft':
                trace_text = sample.get('Text Reasoning Trace[sft]', '')
            elif trace_type == 'textual-cot':
                trace_text = sample.get('Text Reasoning Trace[textual-cot]', '')
            elif trace_type == 'visual-cot':
                trace_text = sample.get('Text Reasoning Trace[visual-cot]', '')
        
        if view_mode == 'plain':
            # Build plain sequence showing exact text and image markers
            plain_sequence = []
            
            # Helper function to process text with image markers
            def process_text_with_images(text, prefix=""):
                sequence = []
                if prefix:
                    sequence.append({'type': 'text', 'content': prefix})
                
                import re
                lines = text.split('\n')
                current_text = ""
                
                for line in lines:
                    # Check for image markers
                    image_match = re.search(r'<image_start>\[(\w+)\]<image_end>', line)
                    if image_match:
                        # Add accumulated text
                        if current_text:
                            sequence.append({'type': 'text', 'content': current_text})
                            current_text = ""
                        
                        # Add image marker and actual image
                        image_ref = image_match.group(1)
                        sequence.append({'type': 'text', 'content': line + '\n'})
                        
                        # Load the actual image if available
                        if image_ref in sample:
                            image_base64 = load_image_as_base64(sample[image_ref])
                            sequence.append({
                                'type': 'image',
                                'name': sample[image_ref],
                                'base64': image_base64
                            })
                    else:
                        current_text += line + '\n'
                
                # Add any remaining text
                if current_text:
                    sequence.append({'type': 'text', 'content': current_text})
                
                return sequence
            
            # Add the question with its images
            question = sample.get('Question', '')
            if question:
                plain_sequence.extend(process_text_with_images(question, "Question: "))
                plain_sequence.append({'type': 'text', 'content': '\n'})
            
            # Add the text reasoning trace with image markers
            if trace_text:
                trace_prefix = f"Text Reasoning Trace [{trace_type_label}]: " if trace_type_label else "Text Reasoning Trace: "
                plain_sequence.extend(process_text_with_images(trace_text, trace_prefix))
                plain_sequence.append({'type': 'text', 'content': '\n'})
            
            # Add final answer
            final_answer = sample.get('Final Answer', '')
            if final_answer:
                plain_sequence.append({'type': 'text', 'content': f"\nFinal Answer: {final_answer}"})
            
            return jsonify({
                'plain_sequence': plain_sequence,
                'available_traces': [{'type': t, 'label': l} for t, l in available_traces],
                'current_trace_type': trace_type,
                'trace_type_label': trace_type_label
            })
        
        else:
            # Rich view (original implementation)
            # Parse the thinking trace
            thoughts = parse_thinking_trace(trace_text) if trace_text else []
            
            # Load images for thoughts
            for i, thought in enumerate(thoughts):
                if 'image_ref' in thought:
                    image_key = thought['image_ref']
                    if image_key in sample:
                        image_base64 = load_image_as_base64(sample[image_key])
                        if image_base64:
                            thought['image'] = image_base64
            
            # Load problem image
            problem_image = None
            if 'problem_image_1' in sample:
                problem_image = load_image_as_base64(sample['problem_image_1'])
            
            # Check if no trace is available or trace is empty
            no_trace = not trace_text or trace_text.strip() == ''
            no_trace_message = None
            if no_trace:
                if trace_type == 'sft':
                    no_trace_message = 'SFT trace is empty (direct answer without reasoning)'
                elif not available_traces:
                    no_trace_message = 'No reasoning trace available for this sample'
                else:
                    no_trace_message = f'{trace_type_label} trace is empty'
            
            return jsonify({
                'question': sample.get('Question', ''),
                'thoughts': thoughts,
                'final_answer': sample.get('Final Answer', ''),
                'problem_image': problem_image,
                'available_traces': [{'type': t, 'label': l} for t, l in available_traces],
                'current_trace_type': trace_type,
                'trace_type_label': trace_type_label,
                'no_trace': no_trace,
                'no_trace_message': no_trace_message
            })
    
    except Exception as e:
        return jsonify({'error': f'Error processing sample: {str(e)}'})

def main():
    """Main function to run the Flask app."""
    global samples, data_dir, jsonl_file
    
    parser = argparse.ArgumentParser(description='Visualize ThinkTrace JSONL files')
    parser.add_argument('--jsonl', required=True, help='Path to JSONL file')
    parser.add_argument('--data-dir', required=True, help='Directory containing images')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the app on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    jsonl_file = args.jsonl
    data_dir = args.data_dir
    
    # Load all samples from JSONL file
    print(f"Loading samples from {jsonl_file}...")
    samples = []
    
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        print(f"Loaded {len(samples)} samples")
        
        if len(samples) == 0:
            print("Error: No samples found in JSONL file")
            return
        
        # Start the Flask app
        print(f"\n🚀 Starting ThinkTrace Visualizer on http://localhost:{args.port}")
        print(f"📁 Data directory: {data_dir}")
        print(f"📊 Total samples: {len(samples)}")
        print("\nPress Ctrl+C to stop the server\n")
        
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
        
    except FileNotFoundError:
        print(f"Error: File not found: {jsonl_file}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL file: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()