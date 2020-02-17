import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorModel(nn.Module):
    def __init__(self, state_space, action_space, parameter_space, init_w=0.003):
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.parameter_space = parameter_space

        self.fc1 = nn.Linear(
            self.state_space.shape[0]+self.parameter_space.shape[0], 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_space.shape[0])

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.state_space.shape[0]+self.parameter_space.shape[0]), 1./np.sqrt(
            self.state_space.shape[0]+self.parameter_space.shape[0]))
        self.fc2.weight.data.uniform_(-1./np.sqrt(400), 1./np.sqrt(400))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out


class CriticModel(nn.Module):
    def __init__(self, state_space, action_space, parameter_space, init_w=0.0003):
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.parameter_space = parameter_space

        self.fc1 = nn.Linear(
            self.state_space.shape[0]+self.parameter_space.shape[0], 400)
        self.fc2 = nn.Linear(400+self.action_space.shape[0], 300)
        self.fc3 = nn.Linear(300, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.state_space.shape[0]+self.parameter_space.shape[0]), 1./np.sqrt(
            self.state_space.shape[0]+self.parameter_space.shape[0]))
        self.fc2.weight.data.uniform_(-1./np.sqrt(
            400+self.action_space.shape[0]), 1./np.sqrt(400+self.action_space.shape[0]))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(torch.cat((out, a), dim=1)))
        out = self.fc3(out)
        return out


class OsiModel(nn.Module):
    def __init__(self, input_space, output_space, init_w=0.003):
        super().__init__()
        self.input_space = input_space
        self.output_space = output_space

        self.fc1 = nn.Linear(self.input_space, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.output_space)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.input_space), 1./np.sqrt(self.input_space))
        self.fc2.weight.data.uniform_(-1./np.sqrt(256), 1./np.sqrt(256))
        self.fc3.weight.data.uniform_(-1./np.sqrt(128), 1./np.sqrt(128))
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = F.tanh(self.fc1(x))
        out = F.tanh(self.fc2(out))
        out = F.tanh(self.fc3(out))
        out = F.tanh(self.fc4(out))
        return out
