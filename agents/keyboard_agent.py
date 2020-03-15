import keyboard
import numpy as np

class KeyboardAgent(object):
    def act(self, state):
        if keyboard.is_pressed('left'):
            return np.array([-.99])
        elif keyboard.is_pressed('right'):
            return np.array([.99])
        else:
            return np.array([0])
