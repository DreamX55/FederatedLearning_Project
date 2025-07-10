Federated Learning for Human Activity Recognition (UCI HAR)

Privacy-Preserving · High-Accuracy · Lightweight PyTorch Pipeline
Overview
Welcome to a robust, privacy-preserving federated learning system for the UCI Human Activity Recognition (HAR) dataset.
Achieve state-of-the-art accuracy while protecting user privacy with differential privacy (DP) applied to model deltas.
All steps are automated and modular, with a lightweight PyTorch backbone.

Project Structure
.
├── data/                   # Raw and processed data
│   └── raw/UCI_HAR/        # UCI HAR dataset (auto-downloaded)
├── results/                # Model checkpoints, logs, tables, figures
├── scripts/                # Pipeline scripts (preprocess, train, validate, evaluate)
├── src/
│   ├── config/             # Training and FL configuration
│   ├── datasets/           # Data loading and preprocessing
│   ├── models/             # Model architectures (FNN)
│   ├── federated/          # Aggregation logic
│   ├── privacy/            # Differential privacy utilities
│   └── evaluation/         # Metrics and analysis
└── run.sh                  # Unified pipeline script


Quickstart
git clone https://github.com/Cypher9802/FederatedLearning_Project.git
cd FederatedLearning_Project
bash run.sh

Everything is automated:
Dataset download, environment setup, preprocessing, training, privacy, and evaluation.


Key Features
Automated End-to-End Pipeline:
One command runs the entire workflow.
Differential Privacy on Deltas:
Industry-standard privacy with minimal accuracy loss.
High Accuracy:
Up to 93% (no privacy) and 91.5–91.7% (strong privacy).
Configurable:
Easily tune rounds, epochs, clients, DP parameters.

Accomplishments
1. Data Pipeline
📥 Automated UCI HAR download and setup.
🔬 Feature normalization to `` with strict clipping.
🧑‍🤝‍🧑 Data split by subject to simulate federated clients.
✅ Integrity and normalization checks.
2. Model
🤖 Feed-Forward Neural Network (FNN):
3 hidden layers (128, 64, 32), ReLU, dropout, modular and extensible.
3. Federated Learning Engine
🏢 Each subject = one client.
🔄 FedAvg aggregation, supports weighted averaging.
⚙️ All parameters centralized for easy tuning.
4. Differential Privacy
🔒 DP on model deltas (client_model - global_model).
🛡️ Configurable noise_multiplier and clip_norm.
📈 Ablation studies for privacy-utility tradeoff.
5. Evaluation & Reporting
📈 Automated accuracy, F1, recall computation.
📑 Results saved in results/ for reproducibility.
📊 Ablation and benchmarking scripts/tables included.
