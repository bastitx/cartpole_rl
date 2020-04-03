import keyboard
import numpy as np

class KeyboardAgent(object):
    def __init__(self, strength):
        self.strength = strength
        
    def act(self, state):
        if keyboard.is_pressed('left'):
            return np.array([-self.strength])
        elif keyboard.is_pressed('right'):
            return np.array([self.strength])
        else:
            return np.array([0])
