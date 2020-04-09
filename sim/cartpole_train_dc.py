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
from scipy.integrate import solve_ivp
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, swingup=True, observe_params=False, solver='rk'):
        self.gravity = 9.81
        self.masscart = 0.43 # kg
        self.masspole = 0.05 # kg
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.13 # actually half the pole's length in meters
        self.polemass_length = (self.masspole * self.length)
        self.mu_cart = 0.0 # friction cart
        self.mu_pole = 0.0003 # friction pole
        self.nc_sign = 1
        self.tau = 0.02 # seconds between state updates
        self.solver = solver

        self.swingup = swingup
        self.observe_params = observe_params

        #add noise?
        #add delay?

        # Angle at which to fail the episode if swingup != False
        self.theta_threshold_radians = 0.21
        self.x_threshold = 0.2 # length of tracks in meters

        high = np.array([self.x_threshold,
                         np.finfo(np.float32).max,
                         np.pi,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low, high)

        low_p = np.array([0.2, 0.05, 0.03, 0.8, 0.01])
        high_p = np.array([1.0, 0.5, 1.0, 1.5, 0.05])
        self.param_space = spaces.Box(low_p, high_p)
        low = np.append(low, low_p)
        high = np.append(high, high_p)
        self.param_observation_space = spaces.Box(low, high)

        if self.observe_params:
            self.observation_space = self.param_observation_space

        self.action_space = spaces.Box(np.array([-10]), np.array([10]))

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @property
    def params(self):
        return np.array([self.masscart, self.masspole, self.length, 
                self.mu_cart, self.mu_pole])
    
    @params.setter
    def params(self, val):
        assert(len(val) == 5)
        self.masscart = val[0]
        self.masspole = val[1]
        self.length = val[2]
        self.mu_cart = val[3]
        self.mu_pole = val[4]
    
    def randomize_params(self):
        self.params = self.param_space.sample()

    
    def f(self, t, y, force):
        x_dot = y[1]
        theta = y[2]
        theta_dot = y[3]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        def get_thetaacc():
            temp = (-force - self.polemass_length * theta_dot**2 * \
                (sintheta + self.mu_cart * self.nc_sign * costheta)) / self.total_mass +  \
                self.mu_cart * self.gravity * self.nc_sign
            return (self.gravity * sintheta + costheta * temp - \
                self.mu_pole * theta_dot / self.polemass_length) / \
                (self.length * (4.0/3.0 - self.masspole * costheta / self.total_mass * \
                (costheta - self.mu_cart * self.nc_sign)))
        
        thetaacc = get_thetaacc()
        nc = self.total_mass * self.gravity - self.polemass_length * \
            (thetaacc * sintheta + theta_dot**2 * costheta)
        
        nc_sign = torch.sign(nc * x_dot).detach()
        if (nc_sign != self.nc_sign).any():
            self.nc_sign = nc_sign
            thetaacc = get_thetaacc()

        xacc  = (force + self.polemass_length * (theta_dot**2 * sintheta - thetaacc * costheta) - \
            self.mu_cart * nc * nc_sign) / self.total_mass
        
        return torch.stack([x_dot, xacc, theta_dot, thetaacc])


    def step(self, action):
        assert isinstance(action, torch.Tensor)
        state = self.state.detach()
        x, x_dot, theta, theta_dot = state.T
        force = action[:,0]

        y0 = torch.stack([x, x_dot, theta, theta_dot]).to(device).detach()
        k1 = self.tau * self.f(0, y0, force)
        k2 = self.tau * self.f(self.tau / 2, y0 + k1 / 2, force)
        k3 = self.tau * self.f(self.tau / 2, y0 + k2 / 2, force)
        k4 = self.tau * self.f(self.tau, y0 + k3, force)
        x_, x_dot_, theta_, theta_dot_ = y0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        theta_ = theta_ % (2 * np.pi)
        theta_ = torch.where(theta_ >= np.pi, theta_ - 2*np.pi, theta_)

        done =  (x < -self.x_threshold) | (x > self.x_threshold)
        if not self.swingup:
            done = done | (theta < -self.theta_threshold_radians) \
                    | (theta > self.theta_threshold_radians)

        self.state = torch.stack((x_,x_dot_,theta_,theta_dot_)).T
        
        if self.swingup:
            reward = torch.where(~done, torch.cos(theta)-0.1*x**2-0.1*torch.abs(x_dot)+1.6, 0)
        else:
            reward = torch.where(~done, torch.ones(done.shape).to(device), torch.zeros(done.shape).to(device))
        
        return self.state, reward, done, {}

    def reset(self):
        state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.swingup:
            state[2] = (state[2] + np.pi) % (2*np.pi)
            if state[2] >= np.pi:
                state[2] -= 2*np.pi
        self.state = torch.tensor(state).float().detach()
        self.steps_beyond_done = None
        self.nc_sign = 1
        return np.array(self.state)

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None