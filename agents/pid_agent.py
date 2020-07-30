import numpy as np

class PIDAgent(object):
	def __init__(self, p=1.0, i=0.0, d=0.0, set_point=0):
		self.p = p
		self.i = i
		self.d = d

		self.set_point = set_point
		self.dt = 0.02

		self.reset()
	
	def reset(self):
		self.pterm = 0.0
		self.iterm = 0.0
		self.dterm = 0.0
		self.last_error = 0

	def act(self, state=None):
		value = state[2]
		error = self.set_point - value
		de = error - self.last_error

		self.pterm = error
		self.iterm += error * self.dt
		self.dterm = de / self.dt

		self.last_error = error

		output = self.p * self.pterm + self.i * self.iterm + self.d * self.dterm
		return max(min(output, 1.0), -1.0)
