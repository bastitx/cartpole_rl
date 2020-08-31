import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.action_var = torch.full(action_shape, action_var).to(device) # torch.nn.Parameter(torch.full(action_shape, action_var)).to(device)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        a_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        dist = torch.distributions.MultivariateNormal(a_mean, cov_mat)
        a = dist.sample()
        a_logprob = dist.log_prob(a)
        return a, a_logprob
    
    def evaluate(self, state, action):
        a_mean = self.actor(state)
        a_var = self.action_var.expand_as(a_mean)
        cov_mat = torch.diag_embed(a_var).to(device)
        dist = torch.distributions.MultivariateNormal(a_mean, cov_mat)
        a_logprobs = dist.log_prob(action.unsqueeze(1))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return a_logprobs, state_value.squeeze(1), dist_entropy

class OsiModel(nn.Module):
    def __init__(self, input_shape, output_shape, init_w=0.003):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.fc1 = nn.Linear(self.input_shape, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100, self.output_shape)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.input_shape), 1./np.sqrt(self.input_shape))
        self.fc2.weight.data.uniform_(-1./np.sqrt(400), 1./np.sqrt(400))
        self.fc3.weight.data.uniform_(-1./np.sqrt(300), 1./np.sqrt(300))
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def forward(self, x): #add drop outs
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        out = self.fc4(out)
        return out

class DCModel(nn.Module):
    def __init__(self, input_shape, output_shape, init_w=0.003, p=0.00):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        #self.lstm = nn.LSTM(self.input_shape, 100)
        self.fc1 = nn.Linear(self.input_shape, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, self.output_shape)
        self.dropout = nn.Dropout(p)

        self.init_weights(init_w)
        self.reset()

    def init_weights(self, init_w):
        self.fc1.weight.data.uniform_(-1./np.sqrt(self.input_shape), 1./np.sqrt(self.input_shape))
        self.fc2.weight.data.uniform_(-1./np.sqrt(50), 1./np.sqrt(50))
        self.fc3.weight.data.uniform_(-1./np.sqrt(50), 1./np.sqrt(50))
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def reset(self, batch_size=1):
        self.hidden = None# (torch.randn(1, batch_size, 100), torch.randn(1, batch_size, 100))

    def forward(self, x):
        #out, self.hidden = self.lstm(x[None], self.hidden)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = torch.tanh(self.fc4(out)) * 10
        return out
