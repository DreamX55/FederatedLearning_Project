import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from src.datasets.har_dataset import HARDataset
from src.models.fnn import FNN
from src.federated.client import Client
from src.federated.server import Server
from src.federated.aggregation import fedavg
from src.privacy.differential_privacy import add_dp_noise

def main():
    processed_dir = 'data/processed'
    checkpoint_dir = 'results/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client_ids = [fname.split('_')[1] for fname in os.listdir(processed_dir) if fname.endswith('_X.npy')]
    global_model = FNN().to(device)
    server = Server(global_model, config={})
    num_rounds = 10
    clients_per_round = min(5, len(client_ids))
    for rnd in range(num_rounds):
        selected = np.random.choice(client_ids, clients_per_round, replace=False)
        client_states = []
        for cid in selected:
            X = np.load(os.path.join(processed_dir, f'client_{cid}_X.npy'))
            y = np.load(os.path.join(processed_dir, f'client_{cid}_y.npy'))
            dataset = HARDataset(X, y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            model = FNN().to(device)
            model.load_state_dict(server.distribute())
            client = Client(cid, model, loader, device, config={}, privacy_fn=add_dp_noise)
            state_dict = client.train()
            client_states.append(state_dict)
        new_global_state = fedavg(client_states)
        server.global_model.load_state_dict(new_global_state)
        torch.save(server.global_model.state_dict(), os.path.join(checkpoint_dir, f'global_model_round_{rnd}.pt'))
        print(f"Completed round {rnd+1}/{num_rounds}")

if __name__ == "__main__":
    main()
