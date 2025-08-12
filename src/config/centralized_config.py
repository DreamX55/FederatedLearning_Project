# Training configuration for centralized and comparative experiments
EPOCHS = 20
ROUNDS = 30
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# You can also use your existing configs for consistency
# Alternative: use values from your existing training_config.py
# EPOCHS = 20
# ROUNDS = 50  # from FL_CONFIG['num_rounds'] 
# LOCAL_EPOCHS = 20  # from TRAINING_CONFIG['local_epochs']
# BATCH_SIZE = 32  # from TRAINING_CONFIG['batch_size']
# LEARNING_RATE = 0.001  # from TRAINING_CONFIG['learning_rate']
