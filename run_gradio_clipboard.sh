#!/bin/bash
# Run the Gradio clipboard image manager app

echo "🖼️  Gradio Clipboard Image Manager"
echo "=================================="
echo ""
echo "Installing requirements..."
pip install -q gradio pillow numpy

echo ""
echo "Creating test_images folder..."
mkdir -p viz/test_images

echo ""
echo "Starting Gradio app..."
echo "📋 Copy images to clipboard and paste them in the app!"
echo "💾 Images will be saved to: viz/test_images/"
echo ""
echo "The app will open in your browser automatically"
echo "Press Ctrl+C to stop the server"
echo ""

cd viz
python gradio_clipboard_app.py