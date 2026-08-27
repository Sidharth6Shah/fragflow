steps for gpu training:

table of contents:
1. set up the instance
2. ssh into gpu
3. run training scripts or command in gpu terminal


1. Set up the instance
- ensure use the right ssh key to accept on the instance (u can check the ssh keys folder on ur lambda labs site to see which keys u have)
- The key should have a public (visible right there on lambda labs), and a private key (created when the key pair was created, and only shown once). The private key should be downloaded somewhere as a .pem file (for a key called lambda-fragflow, it should be lambda-fragflow.pem).
- move the private key to your local machine's hidden .ssh folder:
  '''mv ~/Downloads/lambda-fragflow.pem ~/.ssh/'''
- set permissions so only u can read it:
  '''chmod 600 ~/.ssh/lambda-fragflow.pem'''
- test the connection by ssh'ing in (note: switch that IP address with whatever ur instance's IP address is):
  '''ssh -i ~/.ssh/lambda-fragflow.pem ubuntu@129.213.94.230'''


2. ssh into the gpu
- '''ssh -i ~/.ssh/lambda-fragflow.pem ubuntu@129.213.94.230'''
- type exit to return to local machine
- 


3. run training
- if the plan is to run commands straight in the terminal (no bash or shell scripting), u can just git clone the full repo, install dependancies, and run the training script directly
- if using shell or bash scripting, the steps above are also required, but can just be done on a script instead of manually line by line in the terminal.
  - In this case, a script based approach is being used (scripts in /training_scripts).
  - first, FROM THE LOCAL MACHINE, copy setup_and_train.sh into the gpu:
    '''scp -i ~/.ssh/lambda-fragflow.pem training_scripts/setup_and_train.sh ubuntu@129.213.94.230:~/'''
  - ssh into the gpu make the script executable, and run the script (it has instructions to git clone the repo, etc):
    '''chmod +x setup_and_train.sh'''
    '''./setup_and_train.sh'''


4. check training progress
- make check_training.sh executable:
  '''chmod +x training_scripts/check_training.sh'''
- run check_training.sh from local machine:
  '''training_scripts/check_training.sh 129.213.94.230'''


5. downloaded trained model weights:
- after trainings done:
  '''training_scripts/download_checkpoint.sh 129.213.94.230 10000'''
- early download at iteration <ITERATION_NUMBER>:
  '''training_scripts/download_checkpoint.sh 129.213.94.230 <ITERATION_NUMBER>'''
- move the downloaded weights to the checkpoints folder:
  '''mv ~/Downloads/checkpoint_10000.pt checkpoints/'''