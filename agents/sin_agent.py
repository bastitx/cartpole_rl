import numpy as np

class SinAgent(object):
    def __init__(self, c=0.01, f0=0.1):
        self.c = c
        self.f0 = f0
        self.reset()

    def fun(self, t):
        return np.cos(2*np.pi*(self.c/2 * t**2 + self.f0 * t))
    
    def reset(self):
        self.t = 0

    def act(self, state=None):
        self.t += 0.02
        return np.array([self.fun(self.t)])


if __name__ == '__main__':
    a = SinAgent()
    from matplotlib import pyplot as plt
    x = np.linspace(0, 10, 500)
    plt.plot(x, np.sign(a.fun(x)))
    plt.show()
