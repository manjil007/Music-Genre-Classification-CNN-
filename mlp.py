import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, n_inputs, hidden_sizes, n_outputs, learning_rate, optimizer):
        """
        Initializes the Multi-Layer Perceptron model.
        :param n_inputs: number of input features
        :param hidden_sizes: ordered LIST of the number of neurons in each hidden layer
        :param n_outputs: number of classes to classify
        :param learning_rate: FLOAT of the learning rate
        :param optimizer: STRING of what kind of optimizer we should use "adam" or "sgd"
        """
        super(MLP, self).__init__()

        layers = []
        layer_sizes = [n_inputs] + hidden_sizes + [n_outputs]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        self.learning_rate = learning_rate
        self.optimizer = self._get_optimizer_from_str(optimizer)

    def forward(self, x):
        return self.layers(x)

    def _get_optimizer_from_str(self, optimizer_str):
        if optimizer_str == 'adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_str == 'sgd':
            return optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")

    # TODO: Implement the train method
    # TODO: Implement the predict method
    # TODO: Add ways to tune hyperparameters
    # TODO: Add regularization (e.g., L1, L2, dropout) and way to change the activation function
