import numpy as np

class SinAgent(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.t = 0
        self.f = 1

    def act(self, state):
        self.t += 0.02
        self.f *= 1.01
        return np.array([np.sin(self.t * self.f * 2 * np.pi)])
