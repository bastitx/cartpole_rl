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
        self.min_action = 0.0
        self.dc = DCMotorSim(DCModel, 'dc_model_old.pkl', num_states)
        self.num_states = num_states

        self.seed()
        self.viewer = None
        self.state = None
    
    def preprocessing(self, action):
        self.state_mem = torch.cat((self.state_mem[1:self.num_states].detach(), self.state))
        self.action_mem = torch.cat((self.action_mem[1:self.num_states].detach(), action))
        if self.initializing or action.abs().item() < self.min_action:
            force = torch.tensor([[0]]).detach()
        else:
            comp_state = torch.cat((self.state_mem.flatten(), self.action_mem.flatten()))
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

if __name__ == '__main__':
    import sys
    from agents.keyboard_agent import KeyboardAgent as Agent
    from sim.dc_sim import DCMotorSim
    from model import DCModel
    from util.io import read_data
    env = CartPoleNNDCEnv(swingup=True, observe_params=False)
    #env.x_threshold = 2.0
    agent = Agent(0.9)
    states, actions = read_data('demonstration__2020-05-19__10-49-45.csv')
    states = torch.tensor(states).detach()
    actions = torch.tensor(actions)[:,None]
    state = env.reset()
    #env.state = states[0,None]
    #state = env.state
    done = False
    try:
        #for action in actions:
        while True:
            env.render()
            action = torch.tensor(agent.act(state))[None].float()
            next_state, _, done, _ = env.step(action)
            #memory += [[i, state[0], state[3], action[0]]]
            state = next_state
    except KeyboardInterrupt:
        print("keyboard")
    finally:
        env.close()
        if len(sys.argv) > 1 and '--write-mem' in sys.argv:
            import csv
            with open('memory.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['i', 'x', 'theta', 'action'])
                writer.writerows(memory)
    sys.exit(0)
