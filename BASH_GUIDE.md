# Bash Commands for GPU Training

Quick reference for the specific bash commands used in FragFlow GPU training workflow.

---

## 1. Connect to Lambda GPU

```bash
ssh ubuntu@129.213.16.135
```

**What it does**: Connects to the remote Lambda GPU server via SSH (Secure Shell)

**Breaking it down**:
- `ssh` = secure shell (remote login)
- `ubuntu` = username on the remote server
- `129.213.16.135` = IP address of Lambda GPU instance

**To exit**: Type `exit` or press `Ctrl+D`

---

## 2. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/fragflow.git
cd fragflow
```

**What it does**: Downloads your code to the Lambda GPU and enters the directory

**Breaking it down**:
- `git clone <URL>` = download repository from GitHub
- `cd fragflow` = change directory into fragflow folder

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**What it does**: Installs all Python packages listed in requirements.txt

---

## 4. Start Training in Background

```bash
nohup python training/train.py > training.log 2>&1 &
```

**What it does**: Runs training in background and saves all output to training.log

**Breaking it down**:
- `nohup` = "no hangup" - process survives if you disconnect from SSH
- `python training/train.py` = the actual command to run
- `>` = redirect output
- `training.log` = file where output is saved
- `2>&1` = redirect errors (2) to same place as normal output (1)
- `&` = run in background (gives you terminal back immediately)

**What happens**: You get your terminal prompt back, but training continues running. You can safely disconnect.

---

## 5. Monitor Training Logs

```bash
tail -f training.log
```

**What it does**: Shows live updates from training.log as training runs

**Breaking it down**:
- `tail` = show last lines of a file
- `-f` = "follow" mode - keep updating as new lines are added

**To exit log viewer**: Press `Ctrl+C` (training keeps running, you just exit the viewer)

**View last 20 lines** (not live):
```bash
tail -20 training.log
```

---

## 6. Check if Training is Still Running

```bash
ps aux | grep "python training/train.py" | grep -v grep
```

**What it does**: Shows if your training process is still running

**Breaking it down**:
- `ps aux` = list all running processes
- `|` = "pipe" - send output to next command
- `grep "python training/train.py"` = filter for lines containing this text
- `grep -v grep` = exclude the grep command itself from results

**Expected output**:
- **If training is running**: Shows a line with `python training/train.py`
- **If training finished**: No output (empty)

---

## 7. List Checkpoint Files

```bash
ls -lh checkpoints/
```

**What it does**: Lists files in checkpoints/ directory with human-readable sizes

**Breaking it down**:
- `ls` = list files
- `-l` = long format (shows size, date, permissions)
- `-h` = human-readable sizes (MB, GB instead of bytes)
- `checkpoints/` = directory to list

**Expected output**:
```
-rw-r--r-- 1 ubuntu ubuntu 245M Jan 15 14:23 checkpoint_500.pt
-rw-r--r-- 1 ubuntu ubuntu 245M Jan 15 16:45 checkpoint_1000.pt
```

---

## 8. Download Checkpoint to Local Machine

**IMPORTANT**: Run this from your LOCAL machine, NOT from Lambda GPU!

If you're still SSH'd into Lambda, type `exit` first.

```bash
scp -i ~/Downloads/lambda-key.pem ubuntu@129.213.16.135:~/fragflow/checkpoints/checkpoint_9500.pt ~/Downloads/
```

**What it does**: Securely copies checkpoint file from Lambda GPU to your local Downloads folder

**Breaking it down**:
- `scp` = secure copy (file transfer over SSH)
- `-i ~/Downloads/lambda-key.pem` = use this SSH key for authentication
- `ubuntu@129.213.16.135` = remote server username and IP
- `:~/fragflow/checkpoints/checkpoint_9500.pt` = path to file on remote server
- `~/Downloads/` = destination on your local machine

**Note**: `~` means your home directory (`/Users/yourusername` on Mac, `/home/username` on Linux)

---

## 9. Verify Downloaded File

```bash
ls -lh ~/Downloads/checkpoint_9500.pt
```

**What it does**: Shows file size to confirm download succeeded

**Expected**: File should be ~200-300 MB

**If file is tiny** (few KB): Download failed, retry scp command

---

## Common Patterns Summary

**Redirect output to file**:
```bash
command > output.txt         # Saves output to file (overwrites)
command >> output.txt        # Appends to file
command > log.txt 2>&1       # Saves both output and errors
```

**Background processes**:
```bash
command &                    # Run in background (ends if you disconnect)
nohup command &              # Run in background (survives disconnect)
```

**Chaining commands**:
```bash
cd fragflow && python train.py    # Run second command only if first succeeds
```

**File paths**:
```bash
~                            # Your home directory
.                            # Current directory
..                           # Parent directory
```

---

## Quick Reference by Task

| Task | Command |
|------|---------|
| Connect to GPU | `ssh ubuntu@<IP>` |
| Navigate to folder | `cd fragflow` |
| Start training | `nohup python training/train.py > training.log 2>&1 &` |
| Watch logs live | `tail -f training.log` |
| Check if running | `ps aux \| grep "python training/train.py" \| grep -v grep` |
| List checkpoints | `ls -lh checkpoints/` |
| Download file | `scp -i <keyfile> ubuntu@<IP>:<remote_path> <local_path>` |
| Exit SSH | `exit` |
