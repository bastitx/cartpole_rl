import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.replay_memory import ReplayMemory, Transition_OSI
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OSI():
    def __init__(self, input_space, output_space, lr=0.0001, batch_size=64, memory_size=10000, epochs=200):
        self.input_space = input_space
        self.output_space = output_space
        self.net = model.OsiModel(input_space, output_space).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size, Transition_OSI)
        self.batch_size = batch_size
        self.epochs = epochs

    def predict(self, state):
        return self.net(torch.tensor(state, device=device).float()).detach().cpu().numpy()

    def remember(self, *args):
        self.memory.push(*args)

    def update(self):
        assert(len(self.memory) > 0 and len(self.memory) % self.batch_size == 0)

        batch = Transition_OSI(*zip(*self.memory.memory))
        osi_state_history_batch = torch.tensor(batch.osi_state, device=device).float().detach()
        actual_mu_batch = torch.tensor(np.concatenate(batch.actual_mu), device=device).float().detach()

        losses = []
        for _ in range(self.epochs):
            loss_epoch = 0
            for i in range(0, len(self.memory), self.batch_size):
                predicted_mu_batch = self.net(osi_state_history_batch[i:i+self.batch_size])
                loss = F.mse_loss(predicted_mu_batch, actual_mu_batch[i:i+self.batch_size])
                loss_epoch += loss.detach()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            losses += [loss_epoch / (len(self.memory) // self.batch_size)]
        return losses

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
