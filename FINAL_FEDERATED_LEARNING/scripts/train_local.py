import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src.datasets.har_dataset import HARDataset
from src.models.fnn import FNN
from src.config.training_config import MODEL_CONFIG, TRAINING_CONFIG

def train_local_model(client_id, X, y, epochs=10):
    """
    Train a local model for a specific client with improved hyperparameters.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with new configuration
    model = FNN(**MODEL_CONFIG).to(device)
    
    # Use CrossEntropyLoss (works better than manual softmax)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Create dataset and dataloader
    dataset = HARDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 2 == 0:  # Print every 2 epochs
            avg_loss = total_loss / len(dataloader)
            print(f"Client {client_id}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

def main():
    processed_dir = 'data/processed'
    os.makedirs('results/checkpoints', exist_ok=True)
    
    # Get all client files
    client_files = [f for f in os.listdir(processed_dir) if f.endswith('_X.npy')]
    
    for file in client_files:
        client_id = file.split('_')[1]
        X = np.load(os.path.join(processed_dir, f'client_{client_id}_X.npy'))
        y = np.load(os.path.join(processed_dir, f'client_{client_id}_y.npy'))
        
        # Train with more epochs
        model = train_local_model(client_id, X, y, epochs=TRAINING_CONFIG['local_epochs'])
        
        # Save model
        torch.save(model.state_dict(), f'results/checkpoints/local_model_{client_id}.pt')
        print(f"✓ Trained and saved local model for client {client_id}")

if __name__ == "__main__":
    main()
