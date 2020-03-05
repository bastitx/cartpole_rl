import numpy as np
import matplotlib.pyplot as plt

class Motor():
	def __init__(self, mass=1.1, radius=0.02):
		self.i = 0 # current
		self.omega = 0 # angular velocity
		self.Psi = 18 # flux
		self.R = 2.78 # resistance
		self.L = 0.0063 # inductance
		self.J_rotor = 0.017 # moment of inertia of motor
		self.abc = (0.01, 0.12, 0.1)
		self.radius = radius
		self.mass_pulley = 0.05 # there are two pulleys, estimate of the mass
		self.J_load = mass * self.radius**2 + 2 * 1 / 2 * self.mass_pulley * self.radius**2
		self.J = self.J_rotor + self.J_load
		self.tau = 0.02 # timestep

	def step(self, u, force):
		i_dot = (- self.Psi * self.omega - self.R * self.i + u) / self.L 
		motor_load = np.sign(self.omega) * (self.abc[2] * self.omega * self.omega + \
			np.sign(self.omega) * self.abc[1] * self.omega + self.abc[0] + force * self.radius)
		omega_dot = (self.Psi * self.i - motor_load) / self.J

		self.i += self.tau * i_dot
		self.omega += self.tau * omega_dot
		return self.omega
	
	def reset(self):
		self.i = 0
		self.omega = 0



if __name__ == '__main__':
	m = Motor()
	n = 2000
	l = np.zeros(n)
	for i in range(n):
		l[i] = m.step(np.sin(i*m.tau)*10, 0) / (2 * np.pi) * 60
		if l[i] == np.nan:
			l[i] = 0
			break
	plt.plot(np.arange(n)*m.tau, l)
	plt.plot(np.arange(n)*m.tau, 10 * np.sin(np.arange(n)*m.tau))
	plt.show()
