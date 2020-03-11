#inspired by https://github.com/nikhilbarhate99/PPO-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.replay_memory import ReplayMemory, Transition_PPO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPOAgent():
    def __init__(self, state_shape, action_shape, ActorCritic, gamma=0.99, lr=0.0001, batch_size=64, epochs=80, memory_size=10000):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.gamma = gamma  # discount
        self.epsilon = 0.2  # clip
        self.lr = lr
        self.batch_size = batch_size
        self.policy = ActorCritic(state_shape, action_shape, 0.1).to(device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = ActorCritic(state_shape, action_shape, 0.25).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.epochs = epochs
        self.memory = ReplayMemory(memory_size, Transition_PPO)


    def act(self, state):
        comp_state = torch.tensor(state, device=device).float()
        action, logprob = self.policy_old(comp_state)
        action = action.detach().cpu().numpy()
        action = np.clip(action, -1., 1.)
        return action, logprob.detach().cpu().numpy()

    def remember(self, *args):
        self.memory.push(*args)

    def update(self):
        assert(len(self.memory) > 0 and len(self.memory) % self.batch_size == 0)
        
        batch = Transition_PPO(*zip(*self.memory.memory))
        state_batch = torch.tensor(np.concatenate(batch.state), device=device).float()
        action_batch = torch.tensor(np.concatenate(batch.action), device=device).float()
        logprob_batch = torch.tensor(np.concatenate(batch.logprob), device=device).float()
        reward_batch = np.concatenate(batch.reward)[:, None]
        done_batch = np.concatenate(batch.done)[:, None].astype(np.float)

        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(reward_batch), reversed(done_batch)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, device=device).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for _ in range(self.epochs):
            for i in range(0, len(self.memory), self.batch_size):
                logprobs, state_values, dist_entropy = self.policy.evaluate(state_batch[i:i+self.batch_size], action_batch[i:i+self.batch_size])
                ratios = torch.exp(logprobs - logprob_batch[i:i+self.batch_size])
                advantages = rewards[i:i+self.batch_size] - state_values
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
                loss = - torch.min(surr1, surr2) + 0.5*torch.nn.functional.mse_loss(state_values, rewards[i:i+self.batch_size]) - 0.01 * dist_entropy
                self.optim.zero_grad()
                loss.mean().backward()
                self.optim.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())

    def load_weights(self, folder, epoch, memory=True):
        if folder is None:
            return

        checkpoint = torch.load('{}/checkpoint-{}.pkl'.format(folder, epoch), map_location=device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_old.load_state_dict(checkpoint['policy'])
        self.optim.load_state_dict(checkpoint['optim'])
        if memory:
            for m in checkpoint['memory']:
                self.memory.push(m.state, m.action, m.logprob, m.next_state, m.reward, m.done)

    def save_model(self, output, epoch):
        torch.save({
            'epoch': epoch,
            'policy': self.policy.state_dict(),
            'optim': self.optim.state_dict(),
            'memory': self.memory.memory
        }, '{}/checkpoint-{}.pkl'.format(output, epoch))
