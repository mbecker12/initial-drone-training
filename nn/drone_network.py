import torch
from torch import nn
import torch.nn.functional as F


class DroneBrain(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        n_inputs = kwargs["n_inputs"]
        n_neurons1 = kwargs["n_neurons1"]
        n_neurons2 = kwargs["n_neurons2"]
        n_neurons3 = kwargs["n_neurons3"]
        n_neurons4 = kwargs["n_neurons4"]
        n_outputs = kwargs["n_outputs"]

        self.lin1 = nn.Linear(n_inputs, n_neurons1)
        self.lin2 = nn.Linear(n_neurons1, n_neurons2)
        self.lin3 = nn.Linear(n_neurons2, n_neurons3)
        self.lin4 = nn.Linear(n_neurons3, n_neurons4)
        self.lin5 = nn.Linear(n_neurons4, n_outputs)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = self.lin5(x)

        return torch.sigmoid(x)
