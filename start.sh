#!/bin/bash
# Quick start script for Linux/Mac

echo "============================================"
echo "Starting Pre-test/Post-test Analysis App"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt --quiet

# Run the Streamlit app
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your default browser."
echo "Press Ctrl+C to stop the server."
echo ""
streamlit run app/main.py
