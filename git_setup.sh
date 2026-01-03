#!/bin/bash

echo "--- Setting up Git Configuration ---"

# 1. User Identity (Prefilled)
# PLEASE UPDATE THE EMAIL BELOW BEFORE RUNNING IF NEEDED
git_name="Othsueh"
git_email="ych930719@gmail.com" 

echo "Setting Name: $git_name"
echo "Setting Email: $git_email"

git config --global user.name "$git_name"
git config --global user.email "$git_email"

# 2. Core Editor (Vim)
echo "Setting default editor to Vim..."
git config --global core.editor "vim"

# 3. Default Branch (Main)
echo "Setting default branch to 'main'..."
git config --global init.defaultBranch main

# 4. Credential Helper (Cache passwords for 2 hour)
echo "Enabling credential caching..."
git config --global credential.helper "cache --timeout=7200"

echo "---------------------------------"
echo "Git setup complete! Current settings:"
git config --list | grep "user\|core\|init"
