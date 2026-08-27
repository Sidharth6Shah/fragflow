#!/bin/bash
# Setup and train on Lambda GPU
# Run this script on the Lambda GPU instance after SSH'ing in

set -e  # Exit on error

echo "=========================================="
echo "FragFlow Lambda GPU Training Setup"
echo "=========================================="

# Clone repository
echo ""
echo "[1/4] Cloning repository..."
if [ -d "fragflow" ]; then
    echo "Directory fragflow already exists, skipping clone"
    cd fragflow
    git pull origin main
else
    git clone https://github.com/sidharth6shah/fragflow.git
    cd fragflow
fi

# Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
pip install -r requirements.txt

# Verify setup
echo ""
echo "[3/4] Verifying installation..."
python -c "import torch; import rdkit; print(f'PyTorch: {torch.__version__}')"

# Start training
echo ""
echo "[4/4] Starting training in background..."
nohup python training/train.py > training.log 2>&1 &

TRAIN_PID=$!
echo ""
echo "=========================================="
echo "Training started!"
echo "=========================================="
echo "Process ID: $TRAIN_PID"
echo ""
echo "To monitor progress:"
echo "  tail -f training.log"
echo ""
echo "To check if training is running:"
echo "  ps aux | grep training/train.py"
echo ""
echo "Training will run for ~10 hours"
echo "You can safely disconnect - training continues in background"
echo "=========================================="
