import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src.datasets.har_dataset import HARDataset
from src.models.fnn import FNN
from src.config.training_config import MODEL_CONFIG, TRAINING_CONFIG, FL_CONFIG
from src.federated.aggregation import federated_averaging
from src.privacy.differential_privacy import add_dp_noise

# Set your desired DP parameters here
NOISE_MULTIPLIER = 0.01  # Adjust for privacy/utility tradeoff
CLIP_NORM = 5.0

def train_client_model(client_id, X, y, global_model_state, epochs=5):
    """
    Train a client model for federated learning.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNN(**MODEL_CONFIG).to(device)
    model.load_state_dict(global_model_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    dataset = HARDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    model.train()
    for epoch in range(epochs):
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    # Apply DP noise to the trained model before sending to server
    model = add_dp_noise(model, noise_multiplier=NOISE_MULTIPLIER, clip_norm=CLIP_NORM)
    return model.state_dict()

def main():
    processed_dir = 'data/processed'
    os.makedirs('results/checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = FNN(**MODEL_CONFIG).to(device)
    client_files = [f for f in os.listdir(processed_dir) if f.endswith('_X.npy')]
    clients_data = {}
    for file in client_files:
        client_id = file.split('_')[1]
        X = np.load(os.path.join(processed_dir, f'client_{client_id}_X.npy'))
        y = np.load(os.path.join(processed_dir, f'client_{client_id}_y.npy'))
        clients_data[client_id] = (X, y)
    for round_num in range(FL_CONFIG['num_rounds']):
        client_models = []
        selected_clients = np.random.choice(
            list(clients_data.keys()), 
            size=min(FL_CONFIG['clients_per_round'], len(clients_data)), 
            replace=False
        )
        for client_id in selected_clients:
            print(f"Applying DP noise to client {client_id} in round {round_num}")
            X, y = clients_data[client_id]
            client_model_state = train_client_model(
                client_id, X, y, global_model.state_dict(), 
                epochs=TRAINING_CONFIG['local_epochs'],
            )
            client_models.append(client_model_state)
        global_model_state = federated_averaging(client_models)
        global_model.load_state_dict(global_model_state)
        torch.save(global_model.state_dict(), f'results/checkpoints/global_model_round_{round_num}.pt')
        print(f"Completed round {round_num+1}/{FL_CONFIG['num_rounds']}")

if __name__ == "__main__":
    main()


# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import numpy as np
# from src.datasets.har_dataset import HARDataset
# from src.models.fnn import FNN
# from src.config.training_config import MODEL_CONFIG, TRAINING_CONFIG, FL_CONFIG
# from src.federated.aggregation import federated_averaging
# from src.privacy.differential_privacy import add_dp_noise


# def train_client_model(client_id, X, y, global_model_state, epochs=5):
#     """
#     Train a client model for federated learning.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Create model and load global weights
#     model = FNN(**MODEL_CONFIG).to(device)
#     model.load_state_dict(global_model_state)
    
#     # Setup training
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(
#         model.parameters(), 
#         lr=TRAINING_CONFIG['learning_rate'],
#         weight_decay=TRAINING_CONFIG['weight_decay']
#     )
    
#     # Create dataset and dataloader
#     dataset = HARDataset(X, y)
#     dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    
#     model.train()
#     for epoch in range(epochs):
#         for batch_data, batch_labels in dataloader:
#             batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(batch_data)
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()
    
#     return model.state_dict()

# def main():
#     processed_dir = 'data/processed'
#     os.makedirs('results/checkpoints', exist_ok=True)
    
#     # Initialize global model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     global_model = FNN(**MODEL_CONFIG).to(device)
    
#     # Get all client files
#     client_files = [f for f in os.listdir(processed_dir) if f.endswith('_X.npy')]
    
#     # Load all client data
#     clients_data = {}
#     for file in client_files:
#         client_id = file.split('_')[1]
#         X = np.load(os.path.join(processed_dir, f'client_{client_id}_X.npy'))
#         y = np.load(os.path.join(processed_dir, f'client_{client_id}_y.npy'))
#         clients_data[client_id] = (X, y)
    
#     # Federated learning rounds
#     for round_num in range(FL_CONFIG['num_rounds']):
#         client_models = []
        
#         # Sample clients for this round
#         selected_clients = np.random.choice(
#             list(clients_data.keys()), 
#             size=min(FL_CONFIG['clients_per_round'], len(clients_data)), 
#             replace=False
#         )
        
#         # Train on selected clients
#         for client_id in selected_clients:
#             X, y = clients_data[client_id]
#             client_model_state = train_client_model(
#                 client_id, X, y, global_model.state_dict(), 
#                 epochs=TRAINING_CONFIG['local_epochs']
#             )
#             client_models.append(client_model_state)
        
#         # Aggregate models
#         global_model_state = federated_averaging(client_models)
#         global_model.load_state_dict(global_model_state)
        
#         # Save global model
#         torch.save(global_model.state_dict(), f'results/checkpoints/global_model_round_{round_num}.pt')
#         print(f"Completed round {round_num+1}/{FL_CONFIG['num_rounds']}")

# if __name__ == "__main__":
#     main()
