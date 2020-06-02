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
from model import DCModel
from sim.dc_sim import DCMotorSim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CartPoleNNDCEnv(CartPoleEnv):
    def __init__(self, swingup=True, observe_params=False, randomize=False, num_states=5):
        super().__init__(swingup, observe_params, randomize)
        self.min_action = 0.7
        self.dc = DCMotorSim(DCModel, 'dc_model_old.pkl', num_states)
        self.num_states = num_states

        self.seed()
        self.viewer = None
        self.state = None
    
    def preprocessing(self, action):
        self.state_mem = torch.cat((self.state_mem[1:5], self.state))
        self.action_mem = torch.cat((self.action_mem[1:5], action))
        if self.initializing or action.item() < 0.7:
            force = torch.tensor([[0]]).detach()
        else:
            comp_state = torch.cat((self.state_mem.flatten().detach(), self.action_mem.flatten().detach()))
            force = self.dc.step(comp_state)[None]
        return force[:,0]
    
    def state_input(self, state):
        x, x_dot, theta, theta_dot, *_ = state.T
        return [x, x_dot, theta, theta_dot]

    def reset(self, variance=0.05):
        state = super().reset(1, variance)
        self.state_mem = torch.zeros((self.num_states,4))
        self.action_mem = torch.zeros((self.num_states,1))
        self.initializing = True
        for _ in range(self.num_states):
            state, _, _, _ = self.step(torch.tensor([[0.0]]).detach())
        self.initializing = False
        return state
