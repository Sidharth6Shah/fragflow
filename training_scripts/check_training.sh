#!/bin/bash
# Check training status on Lambda GPU
# Run this script on your LOCAL machine to monitor remote training

# Usage: ./check_training.sh <LAMBDA_IP>
# Example: ./check_training.sh 150.136.23.45

if [ -z "$1" ]; then
    echo "Error: Lambda GPU IP address required"
    echo ""
    echo "Usage: ./check_training.sh <LAMBDA_IP>"
    echo "Example: ./check_training.sh 150.136.23.45"
    exit 1
fi

LAMBDA_IP=$1
SSH_KEY=~/.ssh/lambda-fragflow.pem

echo "=========================================="
echo "FragFlow Training Status"
echo "=========================================="
echo "Lambda IP: $LAMBDA_IP"
echo ""

# Check if training process is running
echo "[1/4] Checking if training process is running..."
if ssh -i "$SSH_KEY" ubuntu@"$LAMBDA_IP" "pgrep -f 'python training/train.py' > /dev/null"; then
    PID=$(ssh -i "$SSH_KEY" ubuntu@"$LAMBDA_IP" "pgrep -f 'python training/train.py'")
    echo "✓ Training is RUNNING (PID: $PID)"
else
    echo "✗ Training is NOT running"
fi

# Show last 20 lines of training log
echo ""
echo "[2/4] Last 20 lines of training log:"
echo "---"
ssh -i "$SSH_KEY" ubuntu@"$LAMBDA_IP" "tail -20 ~/fragflow/training.log 2>/dev/null || echo 'No training.log found'"
echo "---"

# List checkpoints
echo ""
echo "[3/4] Saved checkpoints:"
ssh -i "$SSH_KEY" ubuntu@"$LAMBDA_IP" "ls -lh ~/fragflow/checkpoints/ 2>/dev/null | tail -n +2 || echo 'No checkpoints yet'"

# Estimate progress
echo ""
echo "[4/4] Progress estimate:"
LAST_ITER=$(ssh -i "$SSH_KEY" ubuntu@"$LAMBDA_IP" "grep -oP 'Iter \K\d+' ~/fragflow/training.log 2>/dev/null | tail -1 || echo '0'")
TOTAL_ITERS=10000
if [ "$LAST_ITER" != "0" ]; then
    PROGRESS=$((LAST_ITER * 100 / TOTAL_ITERS))
    echo "Iteration: $LAST_ITER / $TOTAL_ITERS ($PROGRESS%)"
else
    echo "No progress data available yet"
fi

echo ""
echo "=========================================="
echo ""
echo "To view live logs:"
echo "  ssh -i $SSH_KEY ubuntu@$LAMBDA_IP 'tail -f ~/fragflow/training.log'"
echo ""
