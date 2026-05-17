#!/bin/bash
set -e

echo "Starting Blockscope Server..."
echo "================================"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"

# Create data directory if it doesn't exist
echo "Creating data directory..."
mkdir -p /data/vvm33/BLOCKSCOPE_DATA

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if port 9000 is already in use
if lsof -Pi :9000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "WARNING: Port 9000 is already in use!"
    echo "Existing process:"
    lsof -i :9000
    echo ""
    read -p "Kill existing process and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Killing process on port 9000..."
        kill $(lsof -t -i:9000) 2>/dev/null || true
        sleep 2
    else
        echo "Exiting. Please stop the existing process manually."
        exit 1
    fi
fi

# Start the server
echo "Starting FastAPI server on port 9000..."
echo "Data directory: /data/vvm33/BLOCKSCOPE_DATA"
echo "Access at: http://localhost:9000"
echo "================================"
echo ""
python app.py
