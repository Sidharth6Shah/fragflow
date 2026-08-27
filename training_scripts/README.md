# Training Scripts

3 bash scripts to automate Lambda GPU training.

## Quick Start

setup_and_train.sh:
- initial script copied onto the gpu. It has instructions to clone the repo and run the training script.
- Only run once to kick off training


check_training.sh:
- shows last 20 logs of training - can be used to check status of training or check if done

download_checkpoint.sh:
- download final model weights once training is done