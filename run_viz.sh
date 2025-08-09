#!/bin/bash
# Script to run the Zebra-CoT visualization app

echo "Starting Zebra-CoT Dataset Viewer..."
echo "Access the app at http://localhost:8501"
echo "Press Ctrl+C to stop the server"

cd viz
streamlit run app.py --server.port 8501 --server.headless true