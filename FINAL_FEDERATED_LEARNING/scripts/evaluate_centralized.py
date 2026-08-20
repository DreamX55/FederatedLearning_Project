import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from src.models.fnn import FNN
from src.datasets.har_dataset import load_uci_har_data, normalize_data

def evaluate_centralized():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    data_dir = 'data/raw/UCI_HAR/'
    X, y = load_uci_har_data(data_dir)
    X = normalize_data(X)
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load model
    model = FNN().to(device)
    model.load_state_dict(torch.load('models/centralized_model.pt', map_location=device))
    model.eval()
    
    print("Loaded model from models/centralized_model.pt")
    
    # Evaluate
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    
    print(f"Centralized Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")

if __name__ == "__main__":
    evaluate_centralized()
