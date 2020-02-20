import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.replay_memory import ReplayMemory, Transition_PPO


class PPOAgent():
    def __init__(self, state_shape, action_shape, ActorCritic, gamma=0.99, lr=0.0001, batch_size=64, epochs=80, memory_size=10000):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.gamma = gamma  # discount
        self.epsilon = 0.2  # clip
        self.lr = lr
        self.batch_size = batch_size
        self.policy = ActorCritic(state_shape, action_shape, 0.25)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = ActorCritic(state_shape, action_shape, 0.25)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.epochs = epochs
        self.memory = ReplayMemory(memory_size, Transition_PPO)


    def act(self, state):
        comp_state = torch.tensor(state).float()
        action, logprob = self.policy_old(comp_state)
        #action += self.epsilon * self.random_process.sample()
        #action += np.random.normal(scale=self.epsilon)
        #action = np.clip(action, -1., 1.)
        #self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        return action.detach().numpy(), logprob.detach().numpy()

    def remember(self, *args):
        self.memory.push(*args)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = Transition_PPO(*zip(*self.memory.memory))
        state_batch = torch.tensor(np.concatenate(batch.state)).float()
        action_batch = torch.tensor(np.concatenate(batch.action)).float()
        logprob_batch = torch.tensor(np.concatenate(batch.logprob))
        reward_batch = np.concatenate(batch.reward)[:, None]
        done_batch = np.concatenate(batch.done)[:, None].astype(np.float)

        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(reward_batch), reversed(done_batch)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for _ in range(self.epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(state_batch, action_batch)
            ratios = torch.exp(logprobs - logprob_batch)
            advantages = rewards - state_values
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
            loss = - torch.min(surr1, surr2) + 0.5*torch.nn.functional.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())

    def load_weights(self, folder, epoch, memory=True):
        if folder is None:
            return

        checkpoint = torch.load('{}/checkpoint-{}.pkl'.format(folder, epoch))
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.optim_actor.load_state_dict(checkpoint['optim_actor'])
        self.optim_critic.load_state_dict(checkpoint['optim_critic'])
        if memory:
            for m in checkpoint['memory']:
                self.memory.push(m.state, m.action, m.next_state, m.reward, m.done)

    def save_model(self, output, epoch):
        torch.save({
            'epoch': epoch,
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'optim_actor': self.optim_actor.state_dict(),
            'optim_critic': self.optim_critic.state_dict(),
            'memory': self.memory.memory
        }, '{}/checkpoint-{}.pkl'.format(output, epoch))
