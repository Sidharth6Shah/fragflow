#!/bin/bash
# Download checkpoint from Lambda GPU to local machine
# Run this script on your LOCAL machine (not on Lambda GPU)

# Usage: ./download_checkpoint.sh <LAMBDA_IP> [checkpoint_number]
# Example: ./download_checkpoint.sh 150.136.23.45 10000

if [ -z "$1" ]; then
    echo "Error: Lambda GPU IP address required"
    echo ""
    echo "Usage: ./download_checkpoint.sh <LAMBDA_IP> [checkpoint_number]"
    echo "Example: ./download_checkpoint.sh 150.136.23.45 10000"
    exit 1
fi

LAMBDA_IP=$1
CHECKPOINT_NUM=${2:-10000}  # Default to 10000 if not specified
SSH_KEY=~/.ssh/lambda-fragflow.pem
CHECKPOINT_FILE="checkpoint_${CHECKPOINT_NUM}.pt"

echo "=========================================="
echo "Downloading checkpoint from Lambda GPU"
echo "=========================================="
echo "Lambda IP: $LAMBDA_IP"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "SSH Key: $SSH_KEY"
echo ""

# Check if checkpoint exists on Lambda GPU
echo "[1/3] Checking if checkpoint exists on Lambda GPU..."
if ssh -i "$SSH_KEY" ubuntu@"$LAMBDA_IP" "test -f ~/fragflow/checkpoints/$CHECKPOINT_FILE"; then
    echo "✓ Checkpoint found"
else
    echo "✗ Checkpoint not found: ~/fragflow/checkpoints/$CHECKPOINT_FILE"
    echo ""
    echo "Available checkpoints:"
    ssh -i "$SSH_KEY" ubuntu@"$LAMBDA_IP" "ls -lh ~/fragflow/checkpoints/ 2>/dev/null || echo 'No checkpoints directory found'"
    exit 1
fi

# Download checkpoint
echo ""
echo "[2/3] Downloading checkpoint..."
scp -i "$SSH_KEY" ubuntu@"$LAMBDA_IP":~/fragflow/checkpoints/"$CHECKPOINT_FILE" ~/Downloads/

# Verify download
echo ""
echo "[3/3] Verifying download..."
if [ -f ~/Downloads/"$CHECKPOINT_FILE" ]; then
    FILE_SIZE=$(du -h ~/Downloads/"$CHECKPOINT_FILE" | cut -f1)
    echo "✓ Download successful"
    echo "  Location: ~/Downloads/$CHECKPOINT_FILE"
    echo "  Size: $FILE_SIZE"
else
    echo "✗ Download failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Move checkpoint to project: mv ~/Downloads/$CHECKPOINT_FILE checkpoints/"
echo "  2. Generate molecules: python evaluation/sample.py --checkpoint checkpoints/$CHECKPOINT_FILE"
echo "  3. Compute metrics: python evaluation/metrics.py"
echo ""
