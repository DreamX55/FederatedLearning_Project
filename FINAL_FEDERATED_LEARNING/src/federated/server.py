import copy

class Server:
    def __init__(self, global_model, config):
        self.global_model = global_model
        self.config = config

    def aggregate(self, client_state_dicts):
        # Simple FedAvg aggregation
        avg_state_dict = copy.deepcopy(client_state_dicts[0])
        for key in avg_state_dict:
            for i in range(1, len(client_state_dicts)):
                avg_state_dict[key] += client_state_dicts[i][key]
            avg_state_dict[key] = avg_state_dict[key] / len(client_state_dicts)
        self.global_model.load_state_dict(avg_state_dict)
        return self.global_model.state_dict()

    def distribute(self):
        return self.global_model.state_dict()
