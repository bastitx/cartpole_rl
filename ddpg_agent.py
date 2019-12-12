import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayMemory, Transition
from util import soft_update, hard_update


class DDPGAgent():
    def __init__(self, state_space, action_space, ActorModel, CriticModel, gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                    epsilon_decay=0.9999, learning_rate=0.001, tau=0.001, batch_size=64):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma #discount
        self.epsilon = epsilon #exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.actor = ActorModel(self.state_space, self.action_space)
        self.actor_target = ActorModel(self.state_space, self.action_space)
        self.critic = CriticModel(self.state_space, self.action_space)
        self.critic_target = CriticModel(self.state_space, self.action_space)
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(1000)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
    
    def act(self, state):
        action = np.clip(self.actor(torch.tensor(state).float()).detach().numpy() + np.random.normal(scale=self.epsilon), -1, 1)
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        return action
    
    def remember(self, *args):
        self.memory.push(*args)
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = np.concatenate(batch.state)
        action_batch = np.concatenate(batch.action)
        reward_batch = np.concatenate(batch.reward)[:,None]
        next_state_batch = np.concatenate(batch.next_state)
        done_batch = np.concatenate(batch.done)[:,None]

        with torch.no_grad():
            next_q_values = self.critic_target(torch.cat((torch.tensor(next_state_batch).float(), self.actor_target(torch.tensor(next_state_batch).float())), dim=1))

        q_target_batch = torch.tensor(reward_batch).float() + self.gamma * torch.tensor(done_batch.astype(np.float)).float() * next_q_values
        
        self.critic.zero_grad()
        q_batch = self.critic(torch.cat((torch.tensor(state_batch).float(), torch.tensor(action_batch).float()), dim=1))
        value_loss = F.mse_loss(q_batch, q_target_batch)
        value_loss.backward()
        self.optimizer_c.step()

        self.actor.zero_grad()
        policy_loss = -self.critic(torch.cat((torch.tensor(state_batch).float(), self.actor(torch.tensor(state_batch).float())), dim=1))
        policy_loss = policy_loss.mean().backward()
        self.optimizer_a.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        

        
        

