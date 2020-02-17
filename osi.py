import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayMemory, Transition_OSI
import model


class OSI():
    def __init__(self, input_space, output_space, lr=0.001, batch_size=64, memory_size=10000):
        self.input_space = input_space
        self.output_space = output_space
        self.net = model.OsiModel(input_space, output_space)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size, Transition_OSI)
        self.batch_size = batch_size

    def predict(self, states, actions):
        return self.net(torch.cat(torch.tensor(states).float(), torch.tensor(actions).float()))

    def remember(self, *args):
        self.memory.push(*args)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition_OSI(*zip(*transitions))
        state_history_batch = torch.tensor(
            np.concatenate(batch.state_history)).float()
        action_history_batch = torch.tensor(
            np.concatenate(batch.action_history)).float()
        actual_mu_batch = torch.tensor(np.concatenate(batch.actual_mu)).float()

        predicted_mu_batch = self.net(
            torch.cat(state_history_batch, action_history_batch))
        loss = F.mse_loss(predicted_mu_batch, actual_mu_batch)
        loss.backward()
        self.optim.step()

    def load_weights(self, output, epoch, memory=True):
        if output is None:
            return

        checkpoint = torch.load('{}/checkpoint-osi-{}.pkl'.format(output, epoch))
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['optim'])
        if memory:
            for m in checkpoint['memory']:
                self.memory.push(m.state_history, m.action_history, m.actual_mu)

    def save_model(self, output, epoch):
        torch.save({
            'epoch': epoch,
            'net': self.net.state_dict(),
            'optim': self.optim.state_dict(),
            'memory': self.memory.memory
        }, '{}/checkpoint-osi-{}.pkl'.format(output, epoch))
