# GPU Training Guide

Quick guide for training FragFlow on rented GPUs (Lambda Labs).

---

## 1. Launch GPU Instance

**Provider**: [Lambda Labs](https://lambdalabs.com/service/gpu-cloud)

**Recommended setup**:
- GPU: 1x A100 (40GB) - $1.99/hr
- Image: Lambda Stack 22.04 (PyTorch pre-installed)
- Region: Any available

**SSH Key**: Add your public key (`~/.ssh/id_rsa.pub`) in Lambda dashboard before launch.

---

## 2. Connect to Instance

```bash
ssh ubuntu@<INSTANCE_IP>
```

Example: `ssh ubuntu@129.213.16.135`

---

## 3. Setup Repository

```bash
# Clone repo (make sure it's public or use SSH key)
git clone https://github.com/YOUR_USERNAME/fragflow.git
cd fragflow

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Launch Training

```bash
# Start training in background (survives SSH disconnects)
nohup python training/train.py > training.log 2>&1 &

# Monitor logs (Ctrl+C to exit viewer, training continues)
tail -f training.log
```

**Expected output**:
```
Starting training with config: TrainingConfig(...batch_size=16, num_iterations=10000...)
  0%|          | 10/10000 [00:37<9:25:14, 3.39s/it]
```

---

## 5. Monitor Progress

**Check training is running**:
```bash
ps aux | grep "python training/train.py" | grep -v grep
```

**View latest logs**:
```bash
tail -20 training.log
```

**Check iteration speed** (look for `s/it` in progress bar):
- Good: `3-4s/it`
- Slow: `>8s/it` (reduce batch size in `training/config.py`)

**Training time**: ~10 hours for 10K iterations (batch_size=16)

---

## 6. Checkpoints

Models save automatically to `checkpoints/checkpoint_<iter>.pt` every 500 iterations.

**Download checkpoint** (from local machine):
```bash
scp ubuntu@<INSTANCE_IP>:~/fragflow/checkpoints/checkpoint_10000.pt .
```

---

## 7. Terminate Instance

**IMPORTANT**: Stop billing when done!

1. Download all checkpoints first
2. Lambda dashboard → Terminate instance
3. Verify billing stopped

**Cost estimate**: ~$20 for full 10K iteration training

---

## Troubleshooting

**Training too slow (>8s/it)**:
- Edit `training/config.py`: reduce `batch_size` from 16 → 8
- Or reduce `num_iterations` from 10000 → 5000

**RDKit errors in logs**:
- Normal during early training (invalid molecule attempts)
- Model learns to avoid these over time
- Check `avg_reward` increases in logged output

**SSH disconnect**:
- Training continues via `nohup`
- Reconnect with `ssh` and `tail -f training.log`

---

## Config Settings

Current setup (`training/config.py` FULL_CONFIG):
- Vocab: 200 fragments
- Max fragments per molecule: 8
- Batch size: 16 trajectories
- Iterations: 10,000
- Beta: 4.0 (exploration)
- Reward: QED + SA + LogP

Adjust as needed for your use case.
