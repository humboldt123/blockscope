#!/bin/bash
# Deploy Blockscope Server to Brev instance

set -e

BREV_HOST=${1:-vvm33}
REMOTE_DIR="/data/blockscope-server"

echo "================================"
echo "Deploying Blockscope Server to Brev"
echo "================================"
echo "Target host: $BREV_HOST"
echo "Remote directory: $REMOTE_DIR"
echo ""

# Check if we can connect
if ! ssh -o ConnectTimeout=5 "$BREV_HOST" "echo 'Connection successful'" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to $BREV_HOST"
    echo "Please ensure:"
    echo "  1. You have SSH access configured"
    echo "  2. The hostname is correct"
    echo "  3. Your SSH key is added to Brev"
    exit 1
fi

echo "✓ SSH connection verified"
echo ""

# Create remote directory
echo "Creating remote directory..."
ssh "$BREV_HOST" "mkdir -p $REMOTE_DIR"

# Copy files
echo "Copying files to Brev..."
scp -r app.py requirements.txt start.sh README.md "$BREV_HOST:$REMOTE_DIR/"

echo "✓ Files copied successfully"
echo ""

# Make start.sh executable
echo "Setting permissions..."
ssh "$BREV_HOST" "chmod +x $REMOTE_DIR/start.sh"

echo "✓ Permissions set"
echo ""

# Create data directory
echo "Creating data directory..."
ssh "$BREV_HOST" "mkdir -p /data/vvm33/BLOCKSCOPE_DATA"

echo "✓ Data directory created"
echo ""

echo "================================"
echo "Deployment Complete!"
echo "================================"
echo ""
echo "To start the server, run:"
echo "  ssh $BREV_HOST"
echo "  cd $REMOTE_DIR"
echo "  ./start.sh"
echo ""
echo "Or start it remotely:"
echo "  ssh $BREV_HOST 'cd $REMOTE_DIR && nohup ./start.sh > server.log 2>&1 &'"
echo ""
echo "Check status:"
echo "  ssh $BREV_HOST 'curl http://localhost:9000/'"
echo ""
