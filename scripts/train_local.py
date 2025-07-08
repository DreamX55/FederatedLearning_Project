import torch
from torch.utils.data import DataLoader
from src.datasets.har_dataset import HARDataset
from src.models.fnn import FNN
import numpy as np
import os

def main():
    processed_dir = 'data/processed'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fname in os.listdir(processed_dir):
        if fname.endswith('_X.npy'):
            cid = fname.split('_')[1]
            X = np.load(os.path.join(processed_dir, f'client_{cid}_X.npy'))
            y = np.load(os.path.join(processed_dir, f'client_{cid}_y.npy'))
            dataset = HARDataset(X, y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            model = FNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            for epoch in range(10):
                for data, labels in loader:
                    data, labels = data.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            # Save local model
            torch.save(model.state_dict(), f'results/checkpoints/local_model_{cid}.pt')
            print(f"Trained and saved local model for client {cid}")

if __name__ == "__main__":
    main()
