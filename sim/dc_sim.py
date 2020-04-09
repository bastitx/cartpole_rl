import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DCMotorSim():
	def __init__(self, Model, filename=None):
		self.model = Model(5*5, 1).to(device)
		if filename != None:
			self.model.load_state_dict(torch.load(filename, map_location=device))
	
	def step(self, action):
		return self.model(torch.tensor(action, device=device).float()).detach().numpy()

	def train(self, data, env, epochs=10, batch_size=512, lr=0.001):
		optim = torch.optim.Adam(self.model.parameters(), lr=lr)
		states, actions = data
		states = torch.tensor(states, device=device).detach()
		states_mean = states.mean(0)
		states_std = states.std(0)
		states = (states - states_mean) / states_std
		actions = torch.tensor(actions, device=device).detach()
		weights = torch.tensor([1., 0.5, 0.02, 0.0001], device=device).detach()
		for _ in range(epochs):
			epoch_loss = []
			for i in range(5, len(states) - 1 - (len(states) % batch_size), batch_size):
				env.reset()
				env.state = states[i-1:i-1+batch_size]
				res = torch.zeros(batch_size).to(device)
				comp_state = torch.stack([torch.cat((states[j-5:j].flatten(), actions[j-5:j])) for j in range(i, i + batch_size)])
				force = self.model(comp_state)
				state, *_ = env.step(force)
				state = (state - states_mean) / states_std
				res = (states[i:i+batch_size] - state).mv(weights)
				loss = res.pow(2)().mean() # try abs() instead of pow(2)
				#print("Loss: {}".format(loss))
				epoch_loss += [loss.detach()]
				optim.zero_grad()
				loss.backward()
				optim.step()
			print("---------------------\nMean Epoch Loss: {}".format(sum(epoch_loss)/len(epoch_loss)))
		torch.save(self.model.state_dict(), "dc_model.pkl")


if __name__ == "__main__":
	from util.io import read_data
	from sim.cartpole_train_dc import CartPoleEnv
	from model import DCModel
	data = read_data('data.csv')
	sim = DCMotorSim(DCModel, filename=None)
	env = CartPoleEnv(swingup=False)
	sim.train(data, env, 1000, batch_size=128)
