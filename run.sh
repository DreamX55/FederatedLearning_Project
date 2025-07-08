#!/bin/bash
# Usage: bash run_all.sh

# Activate the virtual environment
source .venv/bin/activate

# Preprocess data
python -m scripts.preprocess_data

# Train local models (optional)
python -m scripts.train_local

# Federated training
python -m scripts.train_federated

# Evaluate global model
python -m scripts.evaluate
