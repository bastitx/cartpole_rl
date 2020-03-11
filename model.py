import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorModel(nn.Module):
    def __init__(self, state_shape, action_shape, init_w=0.003):
        super().__init__()
        self.action_shape = action_shape
        self.state_shape = state_shape

        self.fc1 = nn.Linear(self.state_shape[0], 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_shape[0])

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.state_shape[0]), 1./np.sqrt(self.state_shape[0]))
        self.fc2.weight.data.uniform_(-1./np.sqrt(400), 1./np.sqrt(400))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out


class CriticModel(nn.Module):
    def __init__(self, state_shape, action_shape, init_w=0.0003):
        super().__init__()
        self.action_shape = action_shape
        self.state_shape = state_shape

        self.fc1 = nn.Linear(self.state_shape[0], 400)
        self.fc2 = nn.Linear(400+self.action_shape[0], 300)
        self.fc3 = nn.Linear(300, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.state_shape[0]), 1./np.sqrt(self.state_shape[0]))
        self.fc2.weight.data.uniform_(-1./np.sqrt(400+self.action_shape[0]), 1./np.sqrt(400+self.action_shape[0]))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(torch.cat((out, a), dim=1)))
        out = self.fc3(out)
        return out

class CriticModel2(nn.Module):
    def __init__(self, state_shape, action_shape, init_w=0.003):
        super().__init__()
        self.action_shape = action_shape
        self.state_shape = state_shape

        self.fc1 = nn.Linear(self.state_shape[0], 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.state_shape[0]), 1./np.sqrt(self.state_shape[0]))
        self.fc2.weight.data.uniform_(-1./np.sqrt(400), 1./np.sqrt(400))
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    
class ActorCriticModel(nn.Module):
    def __init__(self, state_shape, action_shape, action_var):
        super().__init__()
        self.action_shape = action_shape
        self.state_shape = state_shape

        self.actor = ActorModel(self.state_shape, self.action_shape)
        self.critic = CriticModel2(self.state_shape, self.action_shape)
        self.action_var = torch.nn.Parameter(torch.full(action_shape, action_var))
    
    def forward(self, state):
        a_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var)
        dist = torch.distributions.MultivariateNormal(a_mean, cov_mat)
        a = dist.sample()
        a_logprob = dist.log_prob(a)
        return a, a_logprob
    
    def evaluate(self, state, action):
        a_mean = self.actor(state)
        a_var = self.action_var.expand_as(a_mean)
        cov_mat = torch.diag_embed(a_var)
        dist = torch.distributions.MultivariateNormal(a_mean, cov_mat)
        a_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return a_logprobs, state_value, dist_entropy

class OsiModel(nn.Module):
    def __init__(self, input_shape, output_shape, init_w=0.003):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.fc1 = nn.Linear(self.input_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.output_shape)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.input_shape), 1./np.sqrt(self.input_shape))
        self.fc2.weight.data.uniform_(-1./np.sqrt(256), 1./np.sqrt(256))
        self.fc3.weight.data.uniform_(-1./np.sqrt(128), 1./np.sqrt(128))
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = F.tanh(self.fc1(x))
        out = F.tanh(self.fc2(out))
        out = F.tanh(self.fc3(out))
        out = F.tanh(self.fc4(out))
        return out
