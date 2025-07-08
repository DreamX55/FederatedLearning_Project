import os

import torch
import numpy as np
from src.datasets.har_dataset import HARDataset
from src.models.fnn import FNN
from src.evaluation.metrics import compute_accuracy, compute_f1, compute_recall
from torch.utils.data import DataLoader

def main():
    # Example: Evaluate the latest global model on all clients' data
    processed_dir = 'data/processed'
    model_path = 'results/checkpoints/global_model_round_9.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    for fname in os.listdir(processed_dir):
        if fname.endswith('_X.npy'):
            cid = fname.split('_')[1]
            X = np.load(os.path.join(processed_dir, f'client_{cid}_X.npy'))
            y = np.load(os.path.join(processed_dir, f'client_{cid}_y.npy'))
            dataset = HARDataset(X, y)
            loader = DataLoader(dataset, batch_size=32)
            with torch.no_grad():
                for data, labels in loader:
                    data = data.to(device)
                    outputs = model(data)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
    acc = compute_accuracy(all_labels, all_preds)
    f1 = compute_f1(all_labels, all_preds)
    recall = compute_recall(all_labels, all_preds)
    print(f"Global Model - Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")

if __name__ == "__main__":
    main()
