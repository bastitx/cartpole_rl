import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DCMotorSim():
	def __init__(self, Model):
		self.net = Model(5*5, 1).to(device)
	
	def step(self, state):
		return self.net(state)

	def train(self, data, env, epochs=80, batch_size=512, lr=0.001):
		optim = torch.optim.Adam(self.net.parameters(), lr=lr)
		states, actions = data
		states = torch.tensor(states, device=device)
		actions = torch.tensor(actions, device=device)
		for _ in range(epochs):
			epoch_loss = 0
			for i in range(batch_size-1, len(states) - 1 - len(states) % batch_size, batch_size):
				env.reset()
				env.state = states[i-1].detach()
				res = torch.zeros((batch_size, 4)).to(device)
				for j in range(i, i+batch_size):
					comp_state = torch.cat((states[j-5:j].flatten(), actions[j-5:j]))
					force = self.step(comp_state)
					state, *_ = env.step(force)
					res[j-i] = states[j].detach() - torch.stack(state)
				loss = res.pow(2).mean()
				print("Loss: {}".format(loss))
				epoch_loss += loss.detach()
				optim.zero_grad()
				loss.backward(retain_graph=True)
				optim.step()
			print("---------------------\nEpoch Loss: {}".format(epoch_loss))


if __name__ == "__main__":
	from util.io import read_data
	from sim.cartpole import CartPoleEnv
	import csv
	from model import DCModel
	data = read_data('data.csv')
	env = CartPoleEnv(swingup=False)
	sim = DCMotorSim(DCModel)
	sim.train(data, env, 64)
