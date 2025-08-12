import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.models.fnn import FNN
from src.datasets.har_dataset import load_uci_har_data, normalize_data
from src.config.training_config import TRAINING_CONFIG

def train_centralized():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    data_dir = 'data/raw/UCI_HAR/'
    X, y = load_uci_har_data(data_dir)
    X = normalize_data(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    model = FNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Use same parameters as federated training
    epochs = TRAINING_CONFIG['local_epochs']
    batch_size = TRAINING_CONFIG['batch_size']
    
    # Create data loaders
    train_data = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    print(f"Training centralized model for {epochs} epochs...")
    print(f"Using learning_rate: {TRAINING_CONFIG['learning_rate']}, batch_size: {batch_size}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Training epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/centralized_model.pt')
    print("Centralized model training completed!")

if __name__ == "__main__":
    train_centralized()
