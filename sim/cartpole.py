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

    def __init__(self, swingup=True, observe_params=False, randomize=False, solid_bounds=False):
        self.gravity = 9.81
        self.masscart = 0.43 # kg
        self.masspole = 0.05 # kg
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.13 # actually half the pole's length in meters
        self.polemass_length = (self.masspole * self.length)
        self.mu_cart = 0.0005 # friction cart
        self.mu_pole = 0.0003 # friction pole
        self.nc_sign = 1
        self.tau = 0.02 # seconds between state updates
        self.noise = torch.distributions.normal.Normal(torch.zeros((4,)), torch.tensor([0.01, 0.0001, 0.02, 0.002]))

        self.swingup = swingup
        self.observe_params = observe_params
        self.randomize = randomize
        self.solid_bounds = solid_bounds

        # Angle at which to fail the episode if swingup != False
        self.theta_threshold_radians = 1.57
        self.x_threshold = 0.2 # length of tracks in meters

        high = np.array([self.x_threshold,
                         np.finfo(np.float32).max,
                         np.pi,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low, high)
        # masscart, masspole, length, mu_pole
        low_p = np.array([0.4, 0.03, 0.5, 0.0002])
        high_p = np.array([0.5, 0.07, 0.7, 0.0004])
        self.param_space = spaces.Box(low_p, high_p)
        low = np.append(low, low_p)
        high = np.append(high, high_p)
        self.param_observation_space = spaces.Box(low, high)

        if self.observe_params:
            self.observation_space = self.param_observation_space

        self.action_space = spaces.Box(np.array([-1]), np.array([1]))

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @property
    def params(self):
        return np.array([self.masscart, self.masspole, self.length, self.mu_pole])
    
    @params.setter
    def params(self, val):
        assert(len(val) == 4)
        self.masscart = val[0]
        self.masspole = val[1]
        self.length = val[2]
        self.mu_pole = val[3]
    
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
            (thetaacc * sintheta + theta_dot * theta_dot * costheta)

        nc_sign = torch.sign(nc * x_dot).detach()
        if (nc_sign != self.nc_sign).any():
            self.nc_sign = nc_sign
            thetaacc = get_thetaacc()

        xacc = (force + self.polemass_length * (theta_dot**2 * sintheta - thetaacc * costheta) - \
            self.mu_cart * nc * nc_sign) / self.total_mass
        return torch.stack([x_dot, xacc, theta_dot, thetaacc])
    
    def preprocessing(self, action):
        return action[:,0]
    
    def state_input(self, state):
        x, x_dot, theta, theta_dot, *_ = state.T
        return [x, x_dot, theta, theta_dot]
    
    def state_output(self, new_state):
        return new_state

    def step(self, action):
        assert isinstance(action, torch.Tensor)
        state = self.state.detach()
        u = self.preprocessing(action)

        y0 = torch.stack(self.state_input(state)).to(device).detach()
        k1 = self.tau * self.f(0, y0, u)
        k2 = self.tau * self.f(self.tau / 2, y0 + k1 / 2, u)
        k3 = self.tau * self.f(self.tau / 2, y0 + k2 / 2, u)
        k4 = self.tau * self.f(self.tau, y0 + k3, u)
        x_, x_dot_, theta_, theta_dot_ = self.state_output(y0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

        theta_ = theta_ % (2 * np.pi)
        theta_ = torch.where(theta_ >= np.pi, theta_ - 2*np.pi, theta_)

        if self.solid_bounds:
            x_dot_ = torch.where((x_ > self.x_threshold) | (x_ < -self.x_threshold), torch.zeros_like(x_dot_), x_dot_)
            x_ = torch.where(x_ > self.x_threshold, self.x_threshold * torch.ones_like(x_), x_)
            x_ = torch.where(x_ < -self.x_threshold, -self.x_threshold * torch.ones_like(x_), x_)

        self.state = torch.stack((x_, x_dot_, theta_, theta_dot_)).T

        done =  (x_ < -self.x_threshold) | (x_ > self.x_threshold)
        if not self.swingup:
            done = done | (theta_ < -self.theta_threshold_radians) \
                    | (theta_ > self.theta_threshold_radians)

        reward = torch.where(~done, 12 - theta_**2 - 0.1 * theta_dot_**2 - 0.0001 * torch.abs(u), torch.zeros(done.shape).to(device))

        observation = self.state + self.noise.sample()
        if self.observe_params:
            observation = torch.cat((self.state, torch.stack((*self.params)).T))
        
        return observation, reward, done, {}

    def reset(self, n=1, variance=0.05):
        if self.randomize:
            self.randomize_params()
        state = self.np_random.normal(0, variance, size=(n,4))
        if self.swingup:
            state[:,2] = (state[:,2] + np.pi) % (2*np.pi)
            state[:,2] = np.where(state[:,2] >= np.pi, state[:,2] - 2*np.pi, state[:,2])
        if self.observe_params:
            state = np.append(state, self.params)
        self.state = torch.tensor(state, device=device).float().detach()
        self.nc_sign = torch.ones(n).to(device)
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state[0]
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    import sys
    from agents.keyboard_agent import KeyboardAgent as Agent
    from sim.dc_sim import DCMotorSim
    from model import DCModel
    from util.io import read_data
    env = CartPoleEnv(swingup=True, observe_params=False, solid_bounds=False)
    #env.x_threshold = 2.0
    dc = DCMotorSim(DCModel, 'dc_model_old.pkl', 5)
    agent = Agent(0.9)
    memory = []
    states, actions = read_data('demonstration__2020-05-19__10-49-45.csv')
    states = torch.tensor(states).detach()
    #states_mean = states.mean(0)
    #states_std = states.std(0)
    actions = torch.tensor(actions)[:,None]
    i = 0
    state = env.reset()
    #env.state = states[0,None]
    #state = env.state
    state_mem = torch.zeros((5,4))
    action_mem = torch.zeros((5,1))
    done = False
    try:
        #for action in actions:
        while True:
            env.render()
            #state = (state - states_mean) / states_std
            action = torch.tensor(agent.act(state))[None].float()
            state_mem = torch.cat((state_mem[1:5], state))
            action_mem = torch.cat((action_mem[1:5], action))
            if i >= 5:
                comp_state = torch.cat((state_mem.flatten().detach(), action_mem.flatten().detach()))
                force = dc.step(comp_state)[None]
            else:
                force = torch.tensor([[0]]).detach()
            next_state, _, done, _ = env.step(force)
            #memory += [[i, state[0], state[3], action[0]]]
            state = next_state
            i += 1
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
