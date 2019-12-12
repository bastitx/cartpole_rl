import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayMemory, Transition
from util import soft_update, hard_update
from random_process import OrnsteinUhlenbeckProcess

class DDPGAgent():
    def __init__(self, state_space, action_space, ActorModel, CriticModel, gamma=0.99, epsilon=1.0, epsilon_min=0.1,
                    epsilon_decay=0.9999, lr_actor=0.0001, lr_critic=0.001, tau=0.001, batch_size=64):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma #discount
        self.epsilon = epsilon #exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_space.shape[0], theta=0.15, mu=0, sigma=0.2)
        self.batch_size = batch_size
        self.actor = ActorModel(self.state_space, self.action_space)
        self.actor_target = ActorModel(self.state_space, self.action_space)
        self.critic = CriticModel(self.state_space, self.action_space)
        self.critic_target = CriticModel(self.state_space, self.action_space)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=0.01)
        self.memory = ReplayMemory(600000)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
    
    def act(self, state):
        action = self.actor(torch.tensor(state).float()).detach().numpy()
        #action += self.epsilon * self.random_process.sample()
        action += np.random.normal(scale=self.epsilon)
        action = np.clip(action, -1., 1.)
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        return action
    
    def remember(self, *args):
        self.memory.push(*args)
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(np.concatenate(batch.state)).float()
        action_batch = torch.tensor(np.concatenate(batch.action)).float()
        reward_batch = torch.tensor(np.concatenate(batch.reward)[:,None]).float()
        next_state_batch = np.concatenate(batch.next_state)
        done_batch = torch.tensor(np.concatenate(batch.done)[:,None].astype(np.float)).float()

        with torch.no_grad():
            next_q_values = self.critic_target((torch.tensor(next_state_batch).float(), self.actor_target(torch.tensor(next_state_batch).float())))

        q_target_batch = reward_batch + self.gamma * done_batch * next_q_values
        
        self.critic.zero_grad()
        q_batch = self.critic((state_batch, action_batch))
        value_loss = F.mse_loss(q_batch, q_target_batch)
        value_loss.backward()
        self.optim_critic.step()

        self.actor.zero_grad()
        policy_loss = -self.critic((state_batch, self.actor(state_batch)))
        policy_loss = policy_loss.mean().backward()
        self.optim_actor.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
    
    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

        

        
        

