#!/bin/bash
# Run the Flask visualization app

echo "Starting Zebra-CoT Flask Viewer..."
echo "Installing requirements..."
pip install -q -r viz/requirements_flask.txt

echo ""
echo "Starting server..."
echo "Access the app at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

cd viz
python flask_app.py