import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorModel(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space

        self.fc1 = nn.Linear(self.state_space.shape[0], 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_space.shape[0])
        
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.state_space.shape[0]), 1./np.sqrt(self.state_space.shape[0]))
        self.fc2.weight.data.uniform_(-1./np.sqrt(400), 1./np.sqrt(400))
        self.fc3.weight.data.uniform_(-0.003, 0.003)


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out


class CriticModel(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space

        self.fc1 = nn.Linear(self.state_space.shape[0], 400)
        self.fc2 = nn.Linear(400+self.action_space.shape[0], 300)
        self.fc3 = nn.Linear(300, 1)

        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.state_space.shape[0]), 1./np.sqrt(self.state_space.shape[0]))
        self.fc2.weight.data.uniform_(-1./np.sqrt(400+self.action_space.shape[0]), 1./np.sqrt(400+self.action_space.shape[0]))
        self.fc3.weight.data.uniform_(-0.0003, 0.0003)

    def forward(self, xs):
        x, a = xs
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(torch.cat((out, a), dim=1)))
        out = self.fc3(out)
        return out

