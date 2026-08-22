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

## 6. Check if Training is Finished

### Method 1: Check if process is running
```bash
ps aux | grep "python training/train.py" | grep -v grep
```
- **If you see output** (shows python process) → still running
- **If empty** (no output) → training finished

### Method 2: View latest logs
```bash
tail -20 training.log
```

**What to look for**:

✅ **Training complete** - Look for:
```
100%|██████████| 10000/10000 [9:23:15<00:00, 3.39s/it]
Iter 10000: loss=0.0234, avg_reward=-1.2345, log_Z=5.6789
Saved checkpoint to checkpoints/checkpoint_10000.pt
```

⏳ **Still running** - You'll see:
```
 67%|██████▋   | 6700/10000 [6:23:15<3:14:22, 3.54s/it]
```

⚠️ **RDKit errors** - These are NORMAL during training:
```
[14:32:18] Explicit valence for atom # 0 C, 5, is greater than permitted
```
This means the model is trying invalid molecules. It learns to avoid these over time.

### Method 3: List checkpoints
```bash
ls -lh checkpoints/
```

Expected output when complete:
```
-rw-r--r-- 1 ubuntu ubuntu 245M Jan 15 14:23 checkpoint_500.pt
-rw-r--r-- 1 ubuntu ubuntu 245M Jan 15 16:45 checkpoint_1000.pt
...
-rw-r--r-- 1 ubuntu ubuntu 245M Jan 16 00:12 checkpoint_9500.pt
-rw-r--r-- 1 ubuntu ubuntu 245M Jan 16 00:28 checkpoint_10000.pt
```

**Note**: If training ran for 10K iterations with `save_every=500`, you'll see checkpoints every 500 iterations plus a final checkpoint at 10000.

### Quick verification checklist:
- [ ] `ps aux` shows no python process
- [ ] `tail training.log` shows 100% completion
- [ ] `ls checkpoints/` shows final checkpoint (e.g., checkpoint_10000.pt)
- [ ] File size is ~200-300MB per checkpoint

If all checks pass → training successful! Proceed to download.

---

## 7. After Training Completes

### Step 1: Download Checkpoints

**IMPORTANT**: Run `scp` from your LOCAL machine, NOT the Lambda GPU!

```bash
# From your LOCAL machine (not Lambda GPU):
# If you used SSH key to connect:
scp -i ~/Downloads/lambda-key.pem ubuntu@<INSTANCE_IP>:~/fragflow/checkpoints/checkpoint_10000.pt ~/Downloads/

# Or if using default SSH key:
scp ubuntu@<INSTANCE_IP>:~/fragflow/checkpoints/checkpoint_10000.pt ~/Downloads/

# Download all checkpoints:
scp -i ~/Downloads/lambda-key.pem -r ubuntu@<INSTANCE_IP>:~/fragflow/checkpoints/ ~/Downloads/
```

**Example**:
```bash
scp -i ~/Downloads/lambda-key.pem ubuntu@129.213.16.135:~/fragflow/checkpoints/checkpoint_9500.pt ~/Downloads/
```

**Common mistake**: If you're still SSH'd into Lambda GPU, type `exit` first to return to your local machine.

### Step 2: Verify Download

```bash
# Check file exists and size is reasonable
ls -lh ~/Downloads/checkpoint_9500.pt

# Should show something like:
# -rw-r--r--  1 user  staff   245M Jan 15 14:23 checkpoint_9500.pt
```

**Expected size**: ~200-300 MB per checkpoint

If the file is only a few KB or missing, the download failed - retry the scp command.

### Step 3: (Optional) Generate Samples
```bash
# On Lambda GPU, generate molecules from trained model:
python evaluation/sample.py --checkpoint checkpoints/checkpoint_10000.pt --num_samples 1000

# Download samples to local:
scp -i ~/Downloads/lambda-key.pem ubuntu@<INSTANCE_IP>:~/fragflow/evaluation/sampled_molecules.pkl ~/Downloads/
```

### Step 4: Terminate Instance
**CRITICAL**: Don't forget or you'll keep paying $1.99/hr!

1. **Verify checkpoints downloaded** (see Step 2)
2. **Exit Lambda GPU**: Type `exit` in terminal
3. **Go to Lambda Labs dashboard**: [https://lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)
4. **Navigate**: "Instances" tab
5. **Find your instance**: Look for the running A100 instance
6. **Terminate**: Click "Terminate" button → Confirm
7. **Wait**: Termination takes 5-30 seconds
8. **Verify billing stopped**: Check dashboard shows 0 running instances

**Double-check**: Refresh the page after 1 minute to ensure instance is gone and billing has stopped.

---

## 8. Using Trained Model (Local)

After downloading checkpoint to your local machine:

**Generate molecules**:
```bash
python evaluation/sample.py --checkpoint checkpoint_10000.pt --num_samples 1000
```

**Compute metrics**:
```bash
python evaluation/metrics.py
```

**Expected metrics** (after 10K iterations):
- Validity: 80-90%
- Uniqueness: 40-50%
- Diversity: 75-85%
- Avg reward: increasing over training

**Cost estimate**: ~$20 for full 10K iteration training

---

## 9. Troubleshooting

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

## 10. Config Settings

Current setup (`training/config.py` FULL_CONFIG):
- Vocab: 200 fragments
- Max fragments per molecule: 8
- Batch size: 16 trajectories
- Iterations: 10,000
- Beta: 4.0 (exploration)
- Reward: QED + SA + LogP

Adjust as needed for your use case.
