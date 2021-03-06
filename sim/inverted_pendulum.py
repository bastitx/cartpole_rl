import numpy as np
from gym import utils, spaces
import sim.mujoco_env as mujoco_env
import re
import mujoco_py

XML = '''
<mujoco model="inverted pendulum">
	<compiler inertiafromgeom="true"/>
	<default>
		<joint armature="0" limited="true"/>
		<geom contype="0" rgba="0.7 0.7 0 1"/>
		<tendon/>
	</default>
	<option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
	<size nstack="3000"/>
	<worldbody>
		<geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" friction="1 0.1 0.1" rgba="0.3 0.3 0.7 1" size="0.02 2" type="capsule"/>
		<body name="cart" pos="0 0 0">
			<joint axis="1 0 0" limited="true" name="slider" damping="{slider_damping}" pos="0 0 0" range="-2 2" type="slide"/>
			<geom name="cart" pos="0 0 0" mass="{cart_mass}" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
			<body name="pole" pos="0 0 0">
				<joint axis="0 1 0" name="hinge" damping="{hinge_damping}" pos="0 0 0" limited="false" type="hinge"/>
				<geom fromto="0 0 0 0.001 0 {pole_length}" friction="1 0.1 0.1" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.4" type="capsule" mass="{pole_mass}"/>
			</body>
		</body>
	</worldbody>
	<actuator>
		<general gear="{motor_gear}" gainprm="{motor_gain} 0 0"  joint="slider" name="slide" ctrlrange="-3 3"/>
	</actuator>
</mujoco>
'''
# add frictionloss to joints?
slider_damping = 0.5  # 0
hinge_damping = 0.5  # 1
motor_gear = 200  # 2
motor_gain = 1  # 3
cart_mass = 10  # 4
pole_length = 0.6  # 5
pole_mass = 5  # 6
default_params = np.array([slider_damping, hinge_damping, motor_gear, 
                        motor_gain, cart_mass, pole_length, pole_mass])



class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, swingup=True):
        utils.EzPickle.__init__(self)
        self.swingup = swingup
        self.parameter_space = spaces.Box(np.array(
            [0.2, 0.2, 50, 0.3, 5, 0.1, 1]), np.array([1, 1, 500, 1.5, 25, 2, 20]))
        #self.params = self.parameter_space.sample()
        self.params = default_params
        xml_str = XML.format(
            slider_damping=self.params[0],
            hinge_damping=self.params[1],
            motor_gear=self.params[2],
            motor_gain=self.params[3],
            cart_mass=self.params[4],
            pole_length=self.params[5]*(2*(not swingup)-1),
            pole_mass=self.params[6]
        )
        mujoco_env.MujocoEnv.__init__(self, xml_str, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        if self.swingup:
            ob[1] += np.pi  # starts at bottom
        ob[1] = ob[1] % (2*np.pi)
        if ob[1] >= np.pi:
            ob[1] -= 2*np.pi

        reward = np.cos(ob[1])-0.1*ob[0]**2-0.1*np.abs(ob[3])+1.6

        notdone = np.isfinite(ob).all() and (np.abs(ob[0]) <= 1.95)
        if not self.swingup:
            notdone = notdone and np.cos(ob[1]) > 0.5
        done = not notdone
        if done:
            reward = 0
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + \
            self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reinit(self):
        self.params = self.parameter_space.sample()
        xml_str = XML.format(
            slider_damping=self.params[0],
            hinge_damping=self.params[1],
            motor_gear=self.params[2],
            motor_gain=self.params[3],
            cart_mass=self.params[4],
            pole_length=self.params[5],
            pole_mass=self.params[6]
        )
        self.model = mujoco_py.load_model_from_xml(xml_str)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
