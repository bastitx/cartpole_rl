import numpy as np
import matplotlib.pyplot as plt

class Motor():
	def __init__(self, mass=1.1, radius=0.02):
		self.i = 0 # current
		self.omega = 0 # angular velocity
		self.Psi = 0.08 # flux
		self.R = 1 # resistance
		self.L = 0.05 # inductance
		self.J_rotor = 0.017 # moment of inertia of motor
		self.abc = (0.01, 0.12, 0.1)
		self.radius = radius
		self.mass_pulley = 0.05 # there are two pulleys, estimate of the mass
		self.J_load = mass * self.radius**2 + 2 * 1 / 2 * self.mass_pulley * self.radius**2
		self.J = self.J_rotor #+ self.J_load
		self.tau = 0.02 # timestep

	def step(self, u):
		i_dot = (- self.Psi * self.omega - self.R * self.i + u) / self.L 
		motor_load = np.sign(self.omega) * (self.abc[2] * self.omega**2 + \
			np.sign(self.omega) * self.abc[1] * self.omega + self.abc[0])
		omega_dot = (self.Psi * self.i - motor_load) / self.J

		self.i += self.tau * i_dot
		self.omega += self.tau * omega_dot
		return self.omega

	def reset(self):
		self.i = 0
		self.omega = 0



if __name__ == '__main__':
	m = Motor()
	n = 100
	us = [10, 20, 40, 80]
	for u in us:
		m.reset()
		l = np.zeros(n)
		for i in range(n):
			l[i] = m.step(u) * 60 / (2 * np.pi * m.radius)
		plt.plot(np.arange(n)*m.tau, l)
	#plt.plot(np.arange(n)*m.tau, 10)
	plt.show()
