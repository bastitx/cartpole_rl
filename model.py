import torch
import torch.nn as nn
import numpy as np

class ActorModel(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space

        self.model = nn.Sequential(
            nn.Linear(self.state_space.shape[0], 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.action_space.shape[0]),
            nn.Softsign()
        )

    def forward(self, x):
        return self.model(x)


class CriticModel(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space

        self.model = nn.Sequential(
            nn.Linear(self.state_space.shape[0]+self.action_space.shape[0], 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.model(x)

