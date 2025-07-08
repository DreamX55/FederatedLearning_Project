# FederatedLearning_Project

A modular, reproducible pipeline for federated learning and privacy-preserving activity recognition on the UCI HAR dataset.

## Project Structure

FederatedLearning_Project/
├── data/
│ ├── raw/
│ │ └── UCI_HAR/
│ │ ├── train/
│ │ └── test/
│ └── processed/
├── docs/
├── experiments/
├── results/
├── scripts/
├── src/
├── requirements.txt
└── README.md


## Setup Instructions

1. **Clone the Repository**
git clone https://github.com/Cypher9802/FederatedLearning_Project.git
cd FederatedLearning_Project


2. **Create and Activate a Python Virtual Environment**
python3 -m venv .venv
source .venv/bin/activate


3. **Install Dependencies**
pip install -r requirements.txt


4. **Download and Prepare the UCI HAR Dataset**
- Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
- Unzip the file.
- Move/rename the extracted folder so you have:
  ```
  data/raw/UCI_HAR/
      ├── train/
      │     ├── X_train.txt
      │     ├── y_train.txt
      │     └── subject_train.txt
      └── test/
            ├── X_test.txt
            ├── y_test.txt
            └── subject_test.txt
  ```
- If the folder is named `UCI HAR Dataset`, rename it:
  ```
  mv "data/raw/UCI HAR Dataset" data/raw/UCI_HAR
  ```

## Execution Instructions

**All commands should be run from the project root directory.**

1. **Preprocess the Data**
python -m scripts.preprocess_data

2. **Train Local Models (Optional Baseline)**
python -m scripts.train_local

3. **Federated Training**
python -m scripts.train_federated

4. **Evaluate the Global Model**
python -m scripts.evaluate



## Quick Run Script

You can automate the workflow with:
bash run_all.sh

## Troubleshooting

| Problem                                     | Solution                                                        |
|----------------------------------------------|-----------------------------------------------------------------|
| `ModuleNotFoundError: No module named 'src'`| Run scripts with `python -m scripts.script_name` from root      |
| File not found (e.g., `subject_train.txt`)   | Ensure dataset is in `data/raw/UCI_HAR/` with correct structure |
| Dependency missing                          | Add to `requirements.txt` and reinstall                         |
| Git push rejected                           | `git pull --no-rebase origin main` then resolve and push again  |

## Reproducibility

- All scripts are designed to be run in order for a reproducible pipeline.
- The README and `run_all.sh` ensure new users can replicate your results with minimal setup.

For any issues, please refer to this README or open an issue on the repository.

