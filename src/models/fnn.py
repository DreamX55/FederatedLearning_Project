import torch
import torch.nn as nn

class FNN(nn.Module):
    """
    Feed-Forward Neural Network for UCI HAR activity classification.
    Architecture:
        Input: 561 features
        Hidden Layer 1: 128 neurons, ReLU
        Hidden Layer 2: 64 neurons, ReLU
        Output: 6 classes, Softmax
    """
    def __init__(self, input_dim=561, hidden1=128, hidden2=64, output_dim=6, dropout_p=0.5):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
