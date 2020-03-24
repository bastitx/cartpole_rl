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

    def __init__(self, swingup=True, randomize=False, motortest=False):
        self.gravity = 9.81
        self.masscart = 1.0 # kg
        self.masspole = 0.1 # kg
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length in meters
        self.polemass_length = (self.masspole * self.length)
        self.mu_cart = 0.9 # friction cart
        self.mu_pole = 0.03 # friction pole

        self.nc_sign = 1
        self.i = 0 # current

        self.Psi = 0.1 # flux
        self.R = 1 # resistance
        self.L = 0.05 # inductance
        self.radius = 0.02
        self.J_rotor = 0.017 # moment of inertia of motor
        self.mass_pulley = 0.05 # there are two pulleys, estimate of the mass
        self.J_load = self.total_mass * self.radius**2 + 2 * 1 / 2 * self.mass_pulley * self.radius**2
        self.J = self.J_rotor # should this be J_rotor + J_load or just J_rotor?
        self.tau = 0.02  # seconds between state updates

        self.swingup = swingup
        self.randomize = randomize
        self.motortest = motortest

        low = np.array([0.2, 0.05, 0.03, 0.8, 0.01, 0.01, 0.1, 0.0001])
        high = np.array([1.0, 0.5, 1.0, 1.5, 0.05, 0.5, 5.0, 0.2])
        self.param_space = spaces.Box(low, high, dtype=np.float32)

        if self.randomize:
            self.randomize_params()

        #add noise?

        # Angle at which to fail the episode if swingup != False
        self.theta_threshold_radians = 0.21
        self.x_threshold = 2.4 # length of tracks in meters

        high = np.array([self.x_threshold,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.pi,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

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

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, _, theta, theta_dot, _ = state
        u = action[0]*80

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        def get_thetaacc():
            Tm = self.Psi * self.i
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
        
        i_dot = (-self.Psi * x_dot / self.radius - self.R * self.i + u) / self.L

        thetaacc = get_thetaacc()
        nc = self.total_mass * self.gravity - self.polemass_length * \
            (thetaacc * sintheta + theta_dot * theta_dot * costheta)
        nc_sign = np.sign(nc * x_dot)
        if nc_sign != self.nc_sign:
            self.nc_sign = nc_sign
            thetaacc = get_thetaacc()

        self.xacc = (self.Psi * self.i / self.radius + self.polemass_length * (theta_dot**2 * sintheta - thetaacc * costheta) - \
            self.mu_cart * nc * self.nc_sign) / (self.total_mass + self.J / self.radius**2)

        x  += self.tau * x_dot
        x_dot += self.tau * self.xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        self.i += self.tau * i_dot

        theta = theta % (2*np.pi)
        if theta >= np.pi:
            theta -= 2*np.pi
        self.state = (x,x_dot,self.xacc,theta,theta_dot,thetaacc)

        if self.motortest:
            return x_dot

        done =  x < -self.x_threshold \
                or x > self.x_threshold
        if not self.swingup:
            done = done or theta < -self.theta_threshold_radians \
                    or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = np.cos(theta)-0.1*x**2-0.1*np.abs(x_dot)+1.6
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        if self.swingup:
            self.state[3] = (self.state[3] + np.pi) % (2*np.pi)
            if self.state[3] >= np.pi:
                self.state[3] -= 2*np.pi
        self.i = 0
        self.xacc = 0
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
        self.poletrans.set_rotation(-x[3])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and '--motor-test' in sys.argv:
        import matplotlib.pyplot as plt
        env = CartPoleEnv(swingup=True, randomize=False, motortest=True)
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
        env = CartPoleEnv(swingup=True, randomize=False)
        #env.x_threshold = 20
        agent = Agent()
        memory = []
        i = 0
        state = env.reset()
        done = False
        try:
            while True:
                env.render()
                action = agent.act(state)
                next_state, _, done, _ = env.step(action)
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

