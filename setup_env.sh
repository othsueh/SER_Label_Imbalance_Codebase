#!/bin/bash

echo "--- 1. System Update & Tool Installation ---"
echo "Updating package lists..."
apt update

echo "Installing Vim, Tmux"
apt install -y vim tmux

echo "Verifying system tools:"
vim --version | head -n 1
tmux -V

echo -e "\n--- 2. Python Environment Setup ---"

# 1. Create a python virtual environment named 'exp'
if [ -d "exp" ]; then
    echo "Virtual environment 'exp' already exists."
else
    echo "Creating virtual environment 'exp'..."
    python3 -m venv exp
fi

# 2. Source the environment (Active only for this script execution)
echo "Activating virtual environment..."
source exp/bin/activate

# 3. Install pip-tools inside the virtual environment
echo "Installing pip-tools..."
pip install pip-tools

# 4. Compile requirements.in to requirements.txt
echo "Compiling requirements.in..."
pip-compile requirements.in

# 5. Sync the environment
echo "Syncing packages..."
pip-sync requirements.txt

echo -e "\n--- Setup Complete! ---"
echo "To start working, run this command in your terminal:"
echo "source exp/bin/activate"
