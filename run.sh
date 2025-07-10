#!/bin/bash
# Usage: bash run.sh

set -e  # Exit on error

# --------- CONFIGURATION ---------
UCI_HAR_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
RAW_DATA_DIR="data/raw"
UCI_HAR_DIR="$RAW_DATA_DIR/UCI_HAR"
UCI_HAR_ZIP="UCI_HAR_Dataset.zip"
VENV_DIR=".venv"

# --------- STEP 1: Download UCI HAR Dataset ---------
echo "=== Checking for UCI HAR dataset... ==="
mkdir -p "$RAW_DATA_DIR"

if [ ! -d "$UCI_HAR_DIR" ]; then
    echo "UCI HAR dataset not found. Downloading..."
    wget -O "$RAW_DATA_DIR/$UCI_HAR_ZIP" "$UCI_HAR_URL"
    unzip "$RAW_DATA_DIR/$UCI_HAR_ZIP" -d "$RAW_DATA_DIR"
    # Rename to standard folder name if needed
    if [ -d "$RAW_DATA_DIR/UCI HAR Dataset" ]; then
        mv "$RAW_DATA_DIR/UCI HAR Dataset" "$UCI_HAR_DIR"
    fi
    rm "$RAW_DATA_DIR/$UCI_HAR_ZIP"
    echo "UCI HAR dataset installed at $UCI_HAR_DIR"
else
    echo "UCI HAR dataset already present at $UCI_HAR_DIR"
fi

# --------- STEP 2: Create and Activate Virtual Environment ---------
echo "=== Setting up Python virtual environment... ==="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --------- STEP 3: Install Dependencies ---------
echo "=== Installing Python dependencies... ==="
pip install --upgrade pip
pip install -r requirements.txt

# --------- STEP 4: Run Pipeline Scripts ---------
echo "=== Preprocessing data... ==="
python -m scripts.preprocess_data

echo "=== Validating data... ==="
python -m scripts.data_validation

echo "=== Training local models (optional)... ==="
python -m scripts.train_local

echo "=== Federated training... ==="
python -m scripts.train_federated

echo "=== Evaluating global model... ==="
python -m scripts.evaluate

echo "=== All steps completed successfully! ==="
