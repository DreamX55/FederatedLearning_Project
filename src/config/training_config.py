"""
Training configuration for improved model performance
"""

# Model hyperparameters
MODEL_CONFIG = {
    'input_dim': 561,
    'hidden1': 128,
    'hidden2': 64,
    'hidden3': 32,  # New layer
    'output_dim': 6,
    'dropout_p': 0.3  # Reduced from 0.5 for better performance
}

# Training hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 0.001,  # Reduced from default for better convergence
    'batch_size': 32,
    'local_epochs': 10,  # Increased from 5
    'weight_decay': 1e-4,  # L2 regularization
    'momentum': 0.9
}

# Federated learning configuration
FL_CONFIG = {
    'num_rounds': 20,  # Increased from 10
    'clients_per_round': 10,  # Participate more clients
    'min_clients': 5
}
