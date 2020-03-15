import numpy as np

class PrbsAgent(object):
    def act(self, state):
        return np.array([np.random.choice([-1, 1])])
