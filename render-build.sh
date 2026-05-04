#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "--- Current Directory ---"
pwd
ls -la

echo "--- Directory Structure ---"
ls -R

if [ -f "requirements.txt" ]; then
    echo "--- Installing from root requirements.txt ---"
    pip install -r requirements.txt
elif [ -f "backend/requirements.txt" ]; then
    echo "--- Installing from backend/requirements.txt ---"
    pip install -r backend/requirements.txt
else
    echo "--- ERROR: No requirements.txt found! ---"
    exit 1
fi
