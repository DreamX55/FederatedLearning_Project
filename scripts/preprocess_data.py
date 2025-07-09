import os
import numpy as np
from src.datasets.har_dataset import split_by_subject, augment_data

def main():
    data_dir = 'data/raw/UCI_HAR'
    processed_dir = 'data/processed'
    clients = split_by_subject(data_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    for cid, (X, y) in clients.items():
        X_aug = augment_data(X)
        # Ensure strict normalization after augmentation
        X_aug = np.clip(X_aug, 0, 1)
        np.save(os.path.join(processed_dir, f'client_{cid}_X.npy'), X_aug)
        np.save(os.path.join(processed_dir, f'client_{cid}_y.npy'), y)
    
    print(f"Processed and saved data for {len(clients)} clients.")

if __name__ == "__main__":
    main()
