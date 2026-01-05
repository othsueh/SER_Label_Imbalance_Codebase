#!/bin/bash


echo "--- Setting up Git Configuration ---"

# Load GIT_AUTH_TOKEN from .env or .env.example
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -n "$GIT_AUTH_TOKEN" ]; then
    echo "Loaded GIT_AUTH_TOKEN."
fi


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

if [ -n "$GIT_AUTH_TOKEN" ]; then
    echo "Approving GIT_AUTH_TOKEN for https://github.com..."
    echo "protocol=https
host=github.com
username=$git_name
password=$GIT_AUTH_TOKEN" | git credential approve
fi

echo "---------------------------------"
echo "Git setup complete! Current settings:"
git config --list | grep "user\|core\|init"
