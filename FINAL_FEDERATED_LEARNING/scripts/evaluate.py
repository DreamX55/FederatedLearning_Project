import os
import torch
import torch.nn as nn
import numpy as np
from src.datasets.har_dataset import HARDataset
from src.models.fnn import FNN
from src.evaluation.metrics import compute_accuracy, compute_f1, compute_recall
from src.config.training_config import MODEL_CONFIG, FL_CONFIG
from torch.utils.data import DataLoader

def main():
    # Evaluate the latest global model on all clients' data
    processed_dir = 'data/processed'
    model_path = f'results/checkpoints/global_model_round_{FL_CONFIG["num_rounds"]-1}.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with new configuration
    model = FNN(**MODEL_CONFIG).to(device)
    
    # Load model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using random weights.")
    
    model.eval()
    all_preds, all_labels = [], []
    
    # Get all client files
    client_files = [f for f in os.listdir(processed_dir) if f.endswith('_X.npy')]
    
    for file in client_files:
        client_id = file.split('_')[1]
        X = np.load(os.path.join(processed_dir, f'client_{client_id}_X.npy'))
        y = np.load(os.path.join(processed_dir, f'client_{client_id}_y.npy'))
        
        dataset = HARDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)
        
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(device)
                outputs = model(data)
                # Use torch.argmax for predictions (no softmax in forward pass)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
    
    # Compute metrics
    acc = compute_accuracy(all_labels, all_preds)
    f1 = compute_f1(all_labels, all_preds)
    recall = compute_recall(all_labels, all_preds)
    
    print(f"Global Model - Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")
    
    # Save results to file
    os.makedirs('results/tables', exist_ok=True)
    with open('results/tables/evaluation_results.txt', 'w') as f:
        f.write(f"Global Model Evaluation Results\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Model used: {model_path}\n")

if __name__ == "__main__":
    main()
