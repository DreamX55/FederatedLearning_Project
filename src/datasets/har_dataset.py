import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HARDataset(Dataset):
    """
    PyTorch Dataset for UCI HAR data (feature vectors).
    """
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (np.ndarray): Feature data, shape (num_samples, num_features)
            labels (np.ndarray): Labels, shape (num_samples,)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def load_uci_har_data(data_dir):
    """
    Load UCI HAR dataset from the given directory.
    Assumes data_dir contains 'train' and 'test' folders with X and y files.
    Returns: (X, y) as numpy arrays.
    """
    # Load train data
    X_train = pd.read_csv(os.path.join(data_dir, 'train', 'X_train.txt'), sep=r'\s+', header=None).values
    y_train = pd.read_csv(os.path.join(data_dir, 'train', 'y_train.txt'), sep=r'\s+', header=None).values.flatten() - 1  # zero-based labels

    # Load test data
    X_test = pd.read_csv(os.path.join(data_dir, 'test', 'X_test.txt'), sep=r'\s+', header=None).values
    y_test = pd.read_csv(os.path.join(data_dir, 'test', 'y_test.txt'), sep=r'\s+', header=None).values.flatten() - 1

    # Combine train and test
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    return X, y

def normalize_data(X):
    """
    Normalize data to [0,1] range feature-wise.
    """
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
    return X_norm

def split_by_subject(data_dir):
    """
    Splits the UCI HAR dataset by subject, simulating clients.
    Returns: dict {subject_id: (X, y)}
    """
    # Load subject IDs
    subject_train = pd.read_csv(os.path.join(data_dir, 'train', 'subject_train.txt'), sep=r'\s+', header=None).values.flatten()
    subject_test = pd.read_csv(os.path.join(data_dir, 'test', 'subject_test.txt'), sep=r'\s+', header=None).values.flatten()
    subjects = np.concatenate((subject_train, subject_test))

    X, y = load_uci_har_data(data_dir)
    X = normalize_data(X)

    clients = {}
    for subject_id in np.unique(subjects):
        idx = np.where(subjects == subject_id)[0]
        clients[int(subject_id)] = (X[idx], y[idx])
    return clients

def augment_data(X, noise_std=0.01):
    """
    Simple augmentation: add Gaussian noise to simulate device variation.
    """
    return X + np.random.normal(0, noise_std, X.shape)

# Example usage:
# data_dir = 'data/raw/UCI_HAR/'
# clients = split_by_subject(data_dir)
# client_1_X, client_1_y = clients[1]
# client_1_X_aug = augment_data(client_1_X)
# dataset = HARDataset(client_1_X_aug, client_1_y)

# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# class HARDataset(Dataset):
#     """
#     PyTorch Dataset for UCI HAR data (feature vectors).
#     """
#     def __init__(self, data, labels, transform=None):
#         """
#         Args:
#             data (np.ndarray): Feature data, shape (num_samples, num_features)
#             labels (np.ndarray): Labels, shape (num_samples,)
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.data = torch.tensor(data, dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.long)
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.labels[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample, label

# def load_uci_har_data(data_dir):
#     """
#     Load UCI HAR dataset from the given directory.
#     Assumes data_dir contains 'train' and 'test' folders with X and y files.
#     Returns: (X, y) as numpy arrays.
#     """
#     # Load train data
#     X_train = pd.read_csv(os.path.join(data_dir, 'train', 'X_train.txt'), delim_whitespace=True, header=None).values
#     y_train = pd.read_csv(os.path.join(data_dir, 'train', 'y_train.txt'), delim_whitespace=True, header=None).values.flatten() - 1  # zero-based labels

#     # Load test data
#     X_test = pd.read_csv(os.path.join(data_dir, 'test', 'X_test.txt'), delim_whitespace=True, header=None).values
#     y_test = pd.read_csv(os.path.join(data_dir, 'test', 'y_test.txt'), delim_whitespace=True, header=None).values.flatten() - 1

#     # Combine train and test
#     X = np.vstack((X_train, X_test))
#     y = np.concatenate((y_train, y_test))

#     return X, y

# def normalize_data(X):
#     """
#     Normalize data to [0,1] range feature-wise.
#     """
#     min_vals = X.min(axis=0)
#     max_vals = X.max(axis=0)
#     X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
#     return X_norm

# def split_by_subject(data_dir):
#     """
#     Splits the UCI HAR dataset by subject, simulating clients.
#     Returns: dict {subject_id: (X, y)}
#     """
#     # Load subject IDs
#     subject_train = pd.read_csv(os.path.join(data_dir, 'train', 'subject_train.txt'), delim_whitespace=True, header=None).values.flatten()
#     subject_test = pd.read_csv(os.path.join(data_dir, 'test', 'subject_test.txt'), delim_whitespace=True, header=None).values.flatten()
#     subjects = np.concatenate((subject_train, subject_test))

#     X, y = load_uci_har_data(data_dir)
#     X = normalize_data(X)

#     clients = {}
#     for subject_id in np.unique(subjects):
#         idx = np.where(subjects == subject_id)[0]
#         clients[int(subject_id)] = (X[idx], y[idx])
#     return clients

# def augment_data(X, noise_std=0.01):
#     """
#     Simple augmentation: add Gaussian noise to simulate device variation.
#     """
#     return X + np.random.normal(0, noise_std, X.shape)

# # Example usage:
# # data_dir = 'data/raw/UCI_HAR/'
# # clients = split_by_subject(data_dir)
# # client_1_X, client_1_y = clients[1]
# # client_1_X_aug = augment_data(client_1_X)
# # dataset = HARDataset(client_1_X_aug, client_1_y)
