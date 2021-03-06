#inspired by: https://github.com/ghliu/pytorch-ddpg

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.replay_memory import ReplayMemory, Transition
from util.updates import soft_update, hard_update
from util.random_process import OrnsteinUhlenbeckProcess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent():
    def __init__(self, state_shape, action_shape, ActorModel, CriticModel, gamma=0.99, epsilon=1.0, epsilon_min=0.1,
                 epsilon_decay=0.99995, lr_actor=0.0001, lr_critic=0.001, tau=0.001, batch_size=64, memory_size=10000):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.gamma = gamma  # discount
        self.epsilon = epsilon  # exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        self.random_process = OrnsteinUhlenbeckProcess(
            size=self.action_shape, theta=0.15, mu=0, sigma=0.2)
        self.batch_size = batch_size
        self.actor = ActorModel(self.state_shape, self.action_shape).to(device)
        self.actor_target = ActorModel(self.state_shape, self.action_shape).to(device)
        self.critic = CriticModel(self.state_shape, self.action_shape).to(device)
        self.critic_target = CriticModel(self.state_shape, self.action_shape).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)  # , weight_decay=0.01)
        self.memory = ReplayMemory(memory_size, Transition)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def act(self, state):
        action = self.actor(state).detach()
        #action += torch.tensor(self.epsilon * self.random_process.sample(), device=device).detach()
        action += np.random.normal(scale=self.epsilon)
        action = torch.clamp(action, -1., 1.)
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        return action.squeeze(0).float()

    def remember(self, *args):
        self.memory.push(*args)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1).float()
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        done_batch = torch.cat(batch.done).unsqueeze(1).float()

        with torch.no_grad():
            next_state_batch = torch.cat(batch.next_state)
            next_q_values = self.critic_target((next_state_batch, self.actor_target(next_state_batch)))

        q_target_batch = reward_batch + self.gamma * \
            (1-done_batch) * next_q_values

        self.critic.zero_grad()
        q_batch = self.critic((state_batch, action_batch))
        value_loss = F.mse_loss(q_batch, q_target_batch)
        value_loss.backward()
        self.optim_critic.step()

        self.actor.zero_grad()
        policy_loss = -self.critic((state_batch, self.actor(state_batch))).mean()
        policy_loss.backward()
        self.optim_actor.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def load_weights(self, folder, epoch, memory=True):
        if folder is None:
            return

        checkpoint = torch.load('{}/checkpoint-{}.pkl'.format(folder, epoch), map_location=device)
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
