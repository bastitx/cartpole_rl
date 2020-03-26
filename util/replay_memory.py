from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

Transition_PPO = namedtuple('Transition_PPO', ('state', 'action', 'logprob', 'next_state', 'reward', 'done'))

Transition_OSI = namedtuple('Transition_OSI', ('osi_state', 'actual_mu'))


class ReplayMemory(object):

    def __init__(self, capacity, transition):
        self.capacity = capacity
        self.transition = transition
        self.clear()

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)
