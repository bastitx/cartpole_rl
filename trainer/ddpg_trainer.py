from trainer.trainer import Trainer
import numpy as np
from agents.ddpg_agent import DDPGAgent
import torch

class DDPGTrainer(Trainer):
	def __init__(self, Env, ActorModel, CriticModel):
		super().__init__()
		self.parser.add_argument('--warmup', default=128, type=int)
		self.parser.add_argument('--epsilon', default=1.0, type=float)
		self.parser.add_argument('--epsilon-decay', default=0.9999, type=float)
		self.parser.add_argument('--epsilon-min', default=0.1, type=float)
		self.parser.add_argument('--gamma', default=0.99, type=float)
		self.parser.add_argument('--lr-actor', default=0.0001, type=float)
		self.parser.add_argument('--lr-critic', default=0.001, type=float)
		self.parser.add_argument('--tau', default=0.001, type=float)
		self.parser.add_argument('--batch-size', default=64, type=int)
		self.parser.add_argument('--memory-size', default=10000, type=int)
		self.parser.add_argument('--state-history', default=1, type=int)
		self.args = self.parser.parse_args()
		
		self.init_env(Env)

		state_shape = np.array(self.env.observation_space.shape) * self.args.state_history
		if self.args.observe_params:
			state_shape += np.array(self.env.param_space.shape)
		self.agent = DDPGAgent(tuple(state_shape), self.env.action_space.shape, ActorModel, CriticModel, self.args.gamma, self.args.epsilon, self.args.epsilon_min,
					  self.args.epsilon_decay, self.args.lr_actor, self.args.lr_critic, self.args.tau, self.args.batch_size, self.args.memory_size)
		
		self.load_weights()
	
	def on_done(self):
		self.writer.add_scalar('Epsilon', self.agent.epsilon, self.episode)
		self.writer.add_scalar('Memory', len(self.agent.memory.memory), self.episode)
		self.comp_state = torch.zeros((self.args.state_history, self.env.observation_space.shape[0])).to(self.device)
		self.state = self.env.reset(variance=0.4)
		self.state = self.state.detach()
		self.comp_state = torch.cat((self.state[0,None], self.comp_state[:-1]))
		if self.args.observe_params:
			self.comp_state_ = torch.cat((self.comp_state.flatten(), torch.tensor(self.env.params, device=self.device).float().detach())).unsqueeze(0)
		else:
			self.comp_state_ = self.comp_state.flatten().unsqueeze(0)
	
	def train_step(self, i):
		action = self.agent.act(self.comp_state_)
		next_state, reward, done, _ = self.env.step(action)
		next_state = next_state.detach()
		self.comp_state = torch.cat((next_state[0,None], self.comp_state[:-1]))
		if self.args.observe_params:
			comp_next_state_ = torch.cat((self.comp_state.flatten(), torch.tensor(self.env.params, device=self.device).float().detach())).unsqueeze(0)
		else:
			comp_next_state_ = self.comp_state.flatten().unsqueeze(0)
		self.agent.remember(self.comp_state_, action, comp_next_state_, reward.detach(), done.detach())
		self.comp_state_ = comp_next_state_

		if self.args.mode == 'train' and i >= self.args.warmup:
			self.agent.update()
		
		return next_state, reward, done
