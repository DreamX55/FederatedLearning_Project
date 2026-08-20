import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
from src.models.fnn import FNN
from src.datasets.har_dataset import split_by_subject
from src.config.training_config import FL_CONFIG, TRAINING_CONFIG

def fedavg_aggregate(client_weights):
    """Simple FedAvg aggregation"""
    avg_weights = {}
    for key in client_weights[0].keys():
        avg_weights[key] = torch.stack([client_weights[i][key] for i in range(len(client_weights))]).mean(0)
    return avg_weights

def train_client(model, client_data, client_id, round_num, local_epochs):
    """Train model on client data with detailed logging"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    X, y = client_data
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=True
    )
    
    print(f"Training client {client_id} in round {round_num}")
    
    for epoch in range(local_epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def train_federated_nodp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load client data
    data_dir = 'data/raw/UCI_HAR/'
    clients_data = split_by_subject(data_dir)
    
    # Initialize global model
    global_model = FNN()
    
    # Use EXACT same parameters as FL with DP
    num_rounds = FL_CONFIG['num_rounds']
    clients_per_round = FL_CONFIG['clients_per_round']
    local_epochs = TRAINING_CONFIG['local_epochs']
    
    all_clients = list(clients_data.keys())
    print(f"Training federated model (no DP) with {len(all_clients)} clients for {num_rounds} rounds...")
    print(f"Using {clients_per_round} clients per round, {local_epochs} local epochs")
    
    for round_num in range(num_rounds):
        client_weights = []
        
        # Select same number of clients as DP version
        selected_clients = np.random.choice(
            all_clients, 
            size=min(clients_per_round, len(all_clients)), 
            replace=False
        )
        
        for client_id in selected_clients:
            # Create client model with current global weights
            client_model = FNN()
            client_model.load_state_dict(global_model.state_dict())
            
            # Train on client data
            client_weights.append(train_client(
                client_model, 
                clients_data[client_id], 
                client_id, 
                round_num, 
                local_epochs
            ))
        
        # Aggregate weights
        global_weights = fedavg_aggregate(client_weights)
        global_model.load_state_dict(global_weights)
        
        print(f'Completed round {round_num + 1}/{num_rounds}')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(global_model.state_dict(), 'models/federated_nodp_model.pt')
    print("Federated (no DP) model training completed!")

if __name__ == "__main__":
    train_federated_nodp()
