#!/bin/bash
set -e  # Exit on error

UCI_HAR_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
RAW_DATA_DIR="data/raw"
UCI_HAR_DIR="$RAW_DATA_DIR/UCI_HAR"
UCI_HAR_ZIP="UCI_HAR_Dataset.zip"
VENV_DIR=".venv"

echo "=== Checking for UCI HAR dataset... ==="
mkdir -p "$RAW_DATA_DIR"

if [ ! -d "$UCI_HAR_DIR" ]; then
    echo "UCI HAR dataset not found. Downloading..."

    # Try wget or curl, fail if neither is present
    if command -v wget >/dev/null 2>&1; then
        wget -O "$RAW_DATA_DIR/$UCI_HAR_ZIP" "$UCI_HAR_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$UCI_HAR_URL" -o "$RAW_DATA_DIR/$UCI_HAR_ZIP"
    else
        echo "Error: Neither wget nor curl is installed. Please install one to continue."
        exit 1
    fi

    # Only proceed if the file was downloaded successfully
    if [ ! -f "$RAW_DATA_DIR/$UCI_HAR_ZIP" ]; then
        echo "Error: Failed to download UCI HAR dataset."
        exit 1
    fi

    unzip "$RAW_DATA_DIR/$UCI_HAR_ZIP" -d "$RAW_DATA_DIR"
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

# --------- STEP 4: Run Pipeline Scripts --------
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
