import torch
import sys
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DCMotorSim():
    def __init__(self, Model, filename=None, num_states=5):
        self.num_states = num_states # 4 state + 1 action
        self.model = Model(self.num_states*5, 1, p=0.0).to(device)
        self.model.eval()
        if filename != None:
            self.model.load_state_dict(torch.load(filename, map_location=device))
    
    def step(self, action):
        return self.model(action.detach())

    def train(self, data, env, epochs=10, steps=100, batch_size=512, mini_batch_size=None, batch_step=1, lr=0.001):
        assert(mini_batch_size == None or (batch_size % mini_batch_size == 0 and mini_batch_size < batch_size))
        if mini_batch_size == None:
            mini_batch_size = batch_size
        self.model.train()
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        states, actions = data
        states = torch.tensor(states, device=device).detach()
        #states_mean = states.mean(0)
        #states_std = states.std(0)
        #states = (states - states_mean) / states_std
        actions = torch.tensor(actions, device=device).detach()
        weights = torch.tensor([1., 0.25, 0.0, 0.0000], device=device).detach()
        smallest_loss = np.inf
        for epoch in range(epochs):
            epoch_loss = []
            for i in range(self.num_states, len(states) - steps - (batch_size-1)*batch_step - (len(states) % batch_size), batch_size*batch_step):
                state_bias = torch.normal(torch.zeros((batch_size, 4)), torch.cat((0.4 * torch.ones((batch_size, 1)), torch.zeros((batch_size, 3))), dim=1)).detach()
                env.reset()
                self.model.reset(batch_size)
                env.state = states[i-1:i-1+batch_size*batch_step:batch_step] + state_bias
                res = torch.zeros(batch_size).to(device)
                comp_state = torch.stack([torch.cat(((states[j-self.num_states:j] + state_bias[k]).flatten(), actions[j-self.num_states:j])) for k, j in enumerate(range(i, i + batch_size*batch_step, batch_step))])
                for k in range(steps):
                    force = self.model(comp_state)
                    state, *_ = env.step(force)
                    #state = (state - states_mean) / states_std
                    comp_state = torch.stack([torch.cat((s[1:self.num_states*4], state[j], actions[i+j*batch_step+k-self.num_states+1:i+j*batch_step+k+1])) for j, s in enumerate(comp_state)])
                    res += (states[i+k:i+k+batch_size*batch_step:batch_step] + state_bias - state).mv(weights).pow(2)
                #res = (states[i+steps:i+steps+batch_size*batch_step:batch_step] - state).mv(weights).pow(2)
                for j in range(batch_size // mini_batch_size):
                    loss = res[j*mini_batch_size:(j+1)*mini_batch_size].mean() # try abs() instead of pow(2)
                    #print("Loss: {}".format(loss))
                    epoch_loss += [loss.detach()]
                    optim.zero_grad()
                    if j == batch_size // mini_batch_size - 1:
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)
                    optim.step()
                
            print("---------------------\nMean Episode Loss: {}".format(sum(epoch_loss)/len(epoch_loss)))
            if sum(epoch_loss)/len(epoch_loss) < smallest_loss:
                smallest_loss = sum(epoch_loss)/len(epoch_loss)
                torch.save(self.model.state_dict(), "dc_model_{}.pkl".format(epoch))


if __name__ == "__main__":
    from util.io import read_data
    from sim.cartpole import CartPoleEnv
    from model import DCModel
    data = read_data('demonstration__2020-05-19__10-49-45.csv')
    sim = DCMotorSim(DCModel, 'dc_model_old.pkl', 5)
    env = CartPoleEnv(swingup=True)
    try:
        sim.train(data, env, 1000, batch_size=512, steps=1, mini_batch_size=8, batch_step=1, lr=0.0000001)
    except KeyboardInterrupt:
        torch.save(sim.model.state_dict(), "dc_model_aborted.pkl")
    sys.exit(0)
