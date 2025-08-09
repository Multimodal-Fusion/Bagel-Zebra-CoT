#!/bin/bash
# Run the optimized Flask visualization app

echo "Starting Zebra-CoT Fast Flask Viewer..."
echo "This version loads data on-demand for better performance"
echo ""
echo "Installing requirements..."
pip install -q flask pandas pyarrow pillow numpy

echo ""
echo "Starting server..."
echo "Access the app at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

cd viz
python flask_app_fast.py --port 5000