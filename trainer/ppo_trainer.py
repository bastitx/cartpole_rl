from trainer.trainer import Trainer
import numpy as np
from agents.ppo_agent import PPOAgent
import torch

class PPOTrainer(Trainer):
	def __init__(self, Env, ActorCriticModel):
		super().__init__()
		self.parser.add_argument('--gamma', default=0.99, type=float)
		self.parser.add_argument('--lr', default=0.0001, type=float)
		self.parser.add_argument('--batch-size', default=256, type=int)
		self.parser.add_argument('--memory-size', default=4096, type=int)
		self.parser.add_argument('--epochs', default=16, type=int)
		self.parser.add_argument('--init-variance', default=0.25, type=float)
		self.parser.add_argument('--state-history', default=4, type=int)
		self.args = self.parser.parse_args()
		
		self.init_env(Env)

		state_shape = np.array(self.env.observation_space.shape) * self.args.state_history
		if self.args.observe_params:
			state_shape += np.array(self.env.param_space.shape)
		self.agent = PPOAgent(tuple(state_shape), self.env.action_space.shape, ActorCriticModel, self.args.gamma, 
						self.args.lr, self.args.batch_size, self.args.epochs, self.args.memory_size, self.args.init_variance)
					
		self.load_weights()
	
	def on_done(self):
		self.writer.add_scalar('Variance', self.agent.policy.action_var, self.episode)
		self.writer.add_scalar('Memory', len(self.agent.memory.memory), self.episode)
		#writer.add_scalar('StateValue', agent.policy_old.critic(state), episode)
		self.comp_state = torch.zeros((self.args.state_history, self.env.observation_space.shape[0])).to(self.device)
		self.state = self.env.reset(variance=0.02)
		self.state = self.state.detach()
	
	def train_step(self, i):
		self.comp_state = torch.cat((self.state[0,None], self.comp_state[:-1]))
		if self.args.observe_params:
			comp_state_ = torch.cat((self.comp_state.flatten(), torch.tensor(self.env.params, device=self.device).float().detach())).unsqueeze(0)
		else:
			comp_state_ = self.comp_state.flatten().unsqueeze(0)

		action, logprob = self.agent.act(comp_state_)
		next_state, reward, done, _ = self.env.step(action)
		next_state = next_state.detach()
		self.agent.remember(comp_state_, action, logprob, next_state, reward.detach(), done.detach())

		if self.args.mode == 'train' and i % self.args.memory_size == 0 and i > 0:
			avg_loss = self.agent.update()
			self.writer.add_scalar('Loss/Avg', avg_loss, i)
			self.agent.memory.clear()
			if self.args.randomize:
				done = True
		
		return next_state, reward, done
