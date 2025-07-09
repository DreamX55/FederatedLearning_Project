import torch
import torch.nn as nn

class FNN(nn.Module):
    """
    Enhanced Feed-Forward Neural Network for UCI HAR activity classification.
    Architecture based on FL_Planning_Steps.pdf:
    Input: 561 features
    Hidden Layer 1: 128 neurons, ReLU, Dropout
    Hidden Layer 2: 64 neurons, ReLU, Dropout
    Hidden Layer 3: 32 neurons, ReLU, Dropout (Added for better capacity)
    Output: 6 classes, Softmax
    """
    def __init__(self, input_dim=561, hidden1=128, hidden2=64, hidden3=32, output_dim=6, dropout_p=0.3):
        super(FNN, self).__init__()
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)
        
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)
        
        # Second hidden layer to third hidden layer (NEW)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_p)
        
        # Third hidden layer to output layer
        self.fc4 = nn.Linear(hidden3, output_dim)
        # Note: Removed softmax from forward pass - should be handled in loss function
        
    def forward(self, x):
        # First layer
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Third layer (NEW)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        # No softmax here - CrossEntropyLoss handles it
        return x
