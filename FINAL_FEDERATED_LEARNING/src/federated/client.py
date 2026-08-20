import torch

class Client:
    def __init__(self, client_id, model, train_loader, device, config, privacy_fn=None, optimizer_cls=torch.optim.Adam):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.config = config
        self.privacy_fn = privacy_fn  # Function to add DP noise or secure aggregation
        self.optimizer = optimizer_cls(self.model.parameters(), lr=config.get('training.learning_rate', 0.001))

    def train(self, epochs=None):
        self.model.train()
        epochs = epochs or self.config.get('training.epochs', 10)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        # Optionally apply privacy-preserving mechanism to model
        if self.privacy_fn:
            self.privacy_fn(self.model)
        return self.model.state_dict()

    def set_parameters(self, state_dict):
        self.model.load_state_dict(state_dict)
