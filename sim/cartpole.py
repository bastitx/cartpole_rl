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

        self.action_space = spaces.Box(np.array([-1]), np.array([1]))

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
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

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
        if isinstance(force, torch.Tensor):
            nc_sign = torch.sign(nc * x_dot).detach()
        else:
            nc_sign = np.sign(nc * x_dot)
        if nc_sign != self.nc_sign:
            self.nc_sign = nc_sign
            thetaacc = get_thetaacc()

        xacc  = (force + self.polemass_length * (theta_dot**2 * sintheta - thetaacc * costheta) - \
            self.mu_cart * nc * nc_sign) / self.total_mass
        
        if isinstance(force, torch.Tensor):
            return torch.stack([x_dot, xacc, theta_dot, thetaacc])
        else:
            return np.array([x_dot, xacc, theta_dot, thetaacc])


    def step(self, action):
        if not isinstance(action, torch.Tensor):
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
            state = self.state
        else:
            state = self.state.detach()
        x, x_dot, theta, theta_dot, *_ = state
        force = action[0]
        if self.solver == 'euler':
            _, xacc, _, thetaacc = self.f(0, [x, x_dot, theta, theta_dot], force)
            x_ = x + self.tau * x_dot
            x_dot_ = x_dot + self.tau * xacc
            theta_ = theta + self.tau * theta_dot
            theta_dot_ = theta_dot + self.tau * thetaacc
        elif self.solver == 'rk':
            if isinstance(action, torch.Tensor):
                y0 = torch.tensor([x, x_dot, theta, theta_dot], device=device).detach()
            else:
                y0 = np.array([x, x_dot, theta, theta_dot])
            k1 = self.tau * self.f(0, y0, force)
            k2 = self.tau * self.f(self.tau / 2, y0 + k1 / 2, force)
            k3 = self.tau * self.f(self.tau / 2, y0 + k2 / 2, force)
            k4 = self.tau * self.f(self.tau, y0 + k3, force)
            x_, x_dot_, theta_, theta_dot_ = y0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        else:
            res = solve_ivp(self.f, (0, self.tau), [x, x_dot, theta, theta_dot], args=[force], method=self.solver, t_eval=[self.tau], atol=1, rtol=1)
            if not res['success']:
                raise Exception("Problem in integrator: {}".format(res['message']))
            else:
                x_, x_dot_, theta_, theta_dot_ = res['y'][:,-1]

        theta_ = theta_ % (2 * np.pi)

        if theta_ >= np.pi:
            theta_ = theta_ - 2*np.pi
        if self.observe_params:
            self.state = (x_,x_dot_,theta_,theta_dot_, *self.params)
        else:
            self.state = (x_,x_dot_,theta_,theta_dot_)

        done =  x < -self.x_threshold \
                or x > self.x_threshold
        if not self.swingup:
            done = done or theta < -self.theta_threshold_radians \
                    or theta > self.theta_threshold_radians
        done = bool(done)

        if not done and self.swingup:
            reward = np.cos(theta)-0.1*x**2-0.1*np.abs(x_dot)+1.6
        elif not done:
            reward = 1
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        
        if isinstance(action, torch.Tensor):
            self.state = torch.stack(self.state)
            return self.state, reward, done, {}
        else:
            return np.array(self.state), reward, done, {}

    def reset(self):
        state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.swingup:
            state[2] = (state[2] + np.pi) % (2*np.pi)
            if state[2] >= np.pi:
                state[2] -= 2*np.pi
        if self.observe_params:
            self.state = np.append(state, self.params)
        else:
            self.state = state
        self.steps_beyond_done = None
        return np.array(self.state)

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

        x = self.state
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
    env = CartPoleEnv(swingup=True, observe_params=False, solver='rk')
    env.x_threshold  = 1
    dc = DCMotorSim(DCModel, filename='dc_model.pkl')
    agent = Agent(0.9)
    memory = []
    i = 0
    state = env.reset()
    state_mem = np.zeros((5,4))
    action_mem = np.zeros(5)
    done = False
    try:
        while True:
            env.render()
            action = agent.act(state)
            state_mem = np.roll(state_mem, -1, axis=0)
            action_mem = np.roll(action_mem, -1)
            state_mem[-1] = state
            action_mem[-1] = action
            comp_state = np.concatenate((state_mem.flatten(), action_mem))
            force = dc.step(comp_state)
            next_state, _, done, _ = env.step(force)
            memory += [[i, state[0], state[3], action[0]]]
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

