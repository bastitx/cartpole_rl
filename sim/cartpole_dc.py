#Taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
#modified to include friction and swingup
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import torch
from sim.cartpole import CartPoleEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CartPoleDCEnv(CartPoleEnv):
    def __init__(self, swingup=True, observe_params=False, randomize=False):
        super().__init__(swingup, observe_params, randomize)
        self.i = 0 # current
        self.Psi = 2.2087 # flux
        self.R = 20 # resistance measured
        self.L = 0.5228 # inductance
        self.radius = 0.02
        self.J_rotor = 0.017 # moment of inertia of motor
        self.mass_pulley = 0.05 # there are two pulleys, estimate of the mass
        self.J_load = self.total_mass * self.radius**2 + 2 * 1 / 2 * self.mass_pulley * self.radius**2
        self.J = self.J_rotor #self.J_rotor # should this be J_rotor + J_load or just J_rotor?
        self.max_voltage = 4.372 # measured 20V
        self.transform_factor = 2.697
        self.time_delay = 0 # must be integer of time steps
        self.min_action = 0.7

        high = np.array([self.x_threshold,
                         np.finfo(np.float32).max,
                         np.pi,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low, high)
        # masscart, masspole, length, mu_cart, mu_pole, Psi, R, L, transform_factor, time_delay
        low_p = np.array([0.4, 0.03, 0.5, 0.0004, 0.0002, 2.0, 18, 0.4, 2.5, 0])
        high_p = np.array([0.5, 0.07, 0.7, 0.0009, 0.0004, 2.5, 22, 0.6, 2.9, 3])
        self.param_space = spaces.Box(low_p, high_p)
        low = np.append(low, low_p)
        high = np.append(high, high_p)
        self.param_observation_space = spaces.Box(low, high)

        if self.observe_params:
            self.observation_space = self.param_observation_space

        self.action_space = spaces.Box(np.array([-1]), np.array([1]))
    
    @property
    def params(self):
        return np.array([self.masscart, self.masspole, self.length, 
                self.mu_cart, self.mu_pole, self.Psi, self.R, self.L, 
                self.transform_factor, self.time_delay])
    
    @params.setter
    def params(self, val):
        assert(len(val) == 10)
        self.masscart = val[0]
        self.masspole = val[1]
        self.length = val[2]
        self.mu_cart = val[3]
        self.mu_pole = val[4]
        self.Psi = val[5]
        self.R = val[6]
        self.L = val[7]
        self.transform_factor = val[8]
        self.time_delay = int(val[9])
    
    def f(self, t, y, u):
        x_dot = y[1]
        theta = y[2]
        theta_dot = y[3]
        i = y[4]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        def get_thetaacc():
            Tm = self.Psi * i
            a = Tm / self.radius + self.polemass_length * theta_dot**2 * sintheta
            b = self.mu_cart * self.nc_sign * (self.polemass_length * theta_dot**2 * costheta \
                - self.total_mass * self.gravity)
            c = self.gravity * sintheta - self.mu_pole * theta_dot / self.polemass_length \
                + costheta * (- self.polemass_length * theta_dot**2 * (sintheta + \
                self.mu_cart * self.nc_sign * costheta) / self.total_mass + \
                self.mu_cart * self.gravity * self.nc_sign)
            ab = (a + b) / (self.J / self.radius**2 + self.total_mass)
            return (c - Tm * costheta / (self.radius * self.total_mass) + \
                ab * self.J * costheta / (self.radius**2 * self.total_mass)) / (self.length * \
                (4/3 - self.masspole * costheta / self.total_mass * (costheta - self.mu_cart * self.nc_sign \
                + self.J * (self.mu_cart * self.nc_sign * sintheta - costheta) / \
                (self.J + self.radius**2 * self.total_mass))))
        
        i_dot = (-self.Psi * x_dot / self.radius - self.R * i + u) / self.L

        thetaacc = get_thetaacc()
        nc = self.total_mass * self.gravity - self.polemass_length * \
            (thetaacc * sintheta + theta_dot * theta_dot * costheta)

        nc_sign = torch.sign(nc * x_dot).detach()
        if (nc_sign != self.nc_sign).any():
            self.nc_sign = nc_sign
            thetaacc = get_thetaacc()

        xacc = (self.Psi * i / self.radius + self.polemass_length * (theta_dot**2 * sintheta - thetaacc * costheta) - \
            self.mu_cart * nc * self.nc_sign) / (self.total_mass + self.J / self.radius**2)
        return torch.stack([x_dot, xacc, theta_dot, thetaacc, i_dot])
    
    def preprocessing(self, action):
        u = torch.sign(action) * torch.abs(action)**self.transform_factor * self.max_voltage
        u = torch.where(torch.abs(action) < self.min_action, torch.zeros_like(u), u)
        self.delay_buffer = torch.cat((self.delay_buffer[1:], u.unsqueeze(0)))
        return self.delay_buffer[0]
    
    def state_input(self, state):
        x, x_dot, theta, theta_dot, *_ = state.T
        return [x, x_dot, theta, theta_dot, self.i]

    def state_output(self, new_state):
        x_, x_dot_, theta_, theta_dot_, i_ = new_state
        self.i = i_
        return [x_, x_dot_, theta_, theta_dot_]

    def reset(self, n=1, variance=0.05):
        state = super().reset(n, variance)
        self.i = torch.zeros(n).to(device)
        self.delay_buffer = torch.zeros((int(self.time_delay) + 1, n)).to(device)
        return state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and '--motor-test' in sys.argv:
        import matplotlib.pyplot as plt
        env = CartPoleDCEnv(swingup=True, observe_params=False, motortest=True)
        n = 100
        us = [0.5, 0.75, 1] # * 20V
        for u in us:
            env.reset()
            l = np.zeros(n)
            for i in range(n):
                l[i] = env.step([u])
            plt.plot(np.arange(n)*env.tau, l)
        #plt.plot(np.arange(n)*m.tau, 10)
        plt.show()
    else:
        from agents.keyboard_agent import KeyboardAgent as Agent
        env = CartPoleDCEnv(swingup=False, observe_params=False)
        #env.x_threshold = 20
        agent = Agent(0.9)
        memory = []
        i = 0
        state = env.reset()
        done = False
        try:
            while True:
                env.render()
                action = torch.tensor(agent.act(state)[None], device=device).float()
                next_state, _, done, _ = env.step(action)
                #memory += [[i, state[0], state[3], action[0]]]
                state = next_state
                i += 1
        except KeyboardInterrupt:
            env.close()
            if len(sys.argv) > 1 and '--write-mem' in sys.argv:
                import csv
                with open('memory.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(['i', 'x', 'theta', 'action'])
                    writer.writerows(memory)
        sys.exit(0)

