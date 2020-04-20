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

    def __init__(self, swingup=True, observe_params=False, motortest=False, solver='rk'):
        self.gravity = 9.81
        self.masscart = 0.43 # kg
        self.masspole = 0.05 # kg
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.13 # actually half the pole's length in meters
        self.polemass_length = (self.masspole * self.length)
        self.mu_cart = 0.018 # friction cart 
        self.mu_pole = 0.0003 # friction pole
        self.nc_sign = 1
        self.tau = 0.02 # seconds between state updates
        
        self.i = 0 # current
        self.Psi = 1.046 # flux
        self.R = 20 # resistance
        self.L = 0.100 # inductance
        self.radius = 0.02
        self.J_rotor = 0.017 # moment of inertia of motor
        self.mass_pulley = 0.05 # there are two pulleys, estimate of the mass
        self.J_load = self.total_mass * self.radius**2 + 2 * 1 / 2 * self.mass_pulley * self.radius**2
        self.J = self.J_rotor # should this be J_rotor + J_load or just J_rotor?
        
        self.solver = solver
        self.swingup = swingup
        self.observe_params = observe_params
        self.motortest = motortest

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

        low_p = np.array([0.2, 0.05, 0.03, 0.8, 0.000001, 0.01, 0.1, 0.0001])
        high_p = np.array([1.0, 0.5, 1.0, 1.5, 0.05, 0.5, 5.0, 0.2])
        self.param_space = spaces.Box(low_p, high_p)
        low = np.append(low, low_p)
        high = np.append(high, high_p)
        self.param_observation_space = spaces.Box(low, high)

        if self.observe_params:
            self.observation_space = self.param_observation_space

        self.action_space = spaces.Box(np.array([-30]), np.array([30]))

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
                self.mu_cart, self.mu_pole, self.Psi, self.R, self.L])
    
    @params.setter
    def params(self, val):
        assert(len(val) == 8)
        self.masscart = val[0]
        self.masspole = val[1]
        self.length = val[2]
        self.mu_cart = val[3]
        self.mu_pole = val[4]
        self.Psi = val[5]
        self.R = val[6]
        self.L = val[7]
    
    def randomize_params(self):
        self.params = self.param_space.sample()
    
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

    def step(self, action):
        assert isinstance(action, torch.Tensor)
        state = self.state.detach()
        x, x_dot, theta, theta_dot, *_ = state.T
        u = action[:,0] * 30
        y0 = torch.stack([x, x_dot, theta, theta_dot, self.i]).to(device).detach()
        k1 = self.tau * self.f(0, y0, u)
        k2 = self.tau * self.f(self.tau / 2, y0 + k1 / 2, u)
        k3 = self.tau * self.f(self.tau / 2, y0 + k2 / 2, u)
        k4 = self.tau * self.f(self.tau, y0 + k3, u)
        x_, x_dot_, theta_, theta_dot_, i_ = y0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        theta_ = theta_ % (2 * np.pi)
        theta_ = torch.where(theta_ >= np.pi, theta_ - 2*np.pi, theta_)
        self.i = i_

        if self.observe_params:
            self.state = (x_ , x_dot_, theta_, theta_dot_, *self.params)
        else:
            self.state = (x_, x_dot_, theta_, theta_dot_)
        self.state = torch.stack(self.state).T

        if self.motortest:
            return x_dot

        done =  (x < -self.x_threshold) | (x > self.x_threshold)
        if not self.swingup:
            done = done | (theta < -self.theta_threshold_radians) \
                    | (theta > self.theta_threshold_radians)

        if self.swingup:
            reward = torch.where(~done, torch.cos(theta)-0.1*x**2-0.1*torch.abs(x_dot)+1.6, torch.zeros(done.shape).to(device))
        else:
            reward = torch.where(~done, torch.ones(done.shape).to(device), torch.zeros(done.shape).to(device))
        

        return self.state, reward, done, {}

    def reset(self):
        state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.swingup:
            state[2] = (state[2] + np.pi) % (2*np.pi)
            if state[2] >= np.pi:
                state[2] -= 2*np.pi
        if self.observe_params:
            state = np.append(state, self.params)
        self.state = torch.tensor([state]).float().detach()
        self.i = torch.tensor([0.])
        self.nc_sign = 1
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
    if len(sys.argv) > 1 and '--motor-test' in sys.argv:
        import matplotlib.pyplot as plt
        env = CartPoleEnv(swingup=True, observe_params=False, motortest=True)
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
        env = CartPoleEnv(swingup=True, observe_params=False)
        #env.x_threshold = 20
        agent = Agent(0.9)
        memory = []
        i = 0
        state = env.reset()
        done = False
        try:
            while True:
                env.render()
                action = torch.tensor([agent.act(state)]).float()
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

