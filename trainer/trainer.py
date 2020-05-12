import argparse
from torch.utils.tensorboard import SummaryWriter
import torch

class Trainer():
	def __init__(self, parser=None):
		if parser is None:
			self.parser = argparse.ArgumentParser()

		self.parser.add_argument('--mode', default='train', type=str)
		self.parser.add_argument('--max-iterations', default=100000, type=int)
		self.parser.add_argument('--max-episode-length', default=500, type=int)
		self.parser.add_argument('--resume-episode', default=0, type=int)
		self.parser.add_argument('--randomize', dest='randomize', action='store_true')
		self.parser.set_defaults(randomize=False)
		self.parser.add_argument('--render', dest='render', action='store_true')
		self.parser.set_defaults(render=False)
		self.parser.add_argument('--observe-params', dest='observe_params', action='store_true')
		self.parser.set_defaults(render=False)
		self.parser.add_argument('--no-swingup', dest='swingup', action='store_false')
		self.parser.set_defaults(swingup=True)

		self.writer = SummaryWriter()
		self.agent = None
		self.args = None
		self.episode = 0
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def init_env(self, Env):
		self.env = Env(self.args.swingup)
	
	def load_weights(self):
		if self.args.resume_episode > 0:
			try:
				self.agent.load_weights('models', self.args.resume_episode)
				print("loaded weights")
				self.episode = self.args.resume_episode
			except Exception as e:
				print("Couldn't load weights! {}".format(e))
				self.episode = 0
		else:
			self.episode = 0
	
	def train(self):
		done = True
		episode_reward = 0
		last_episode_start = -1
		for i in range(self.args.max_iterations):
			if done:
				self.episode += 1
				episode_length = i - last_episode_start
				self.writer.add_scalar('EpisodeReward/total', episode_reward, self.episode)
				self.writer.add_scalar('EpisodeReward/avg', episode_reward/episode_length, self.episode)
				self.writer.add_scalar('EpisodeLength', episode_length, self.episode)
				if self.args.randomize:
					self.env.randomize_params()
				self.on_done()
				self.writer.flush()
				episode_reward = 0
				last_episode_start = i

			if self.args.render:
				self.env.render()
			
			next_state, reward, done = self.train_step(i)

			episode_reward += reward
			self.state = next_state

			if i - last_episode_start >= self.args.max_episode_length:
				self.done = True

		self.env.close()
		if self.args.mode == 'train':
			self.agent.save_model('models', self.episode)
			print("Saved model: {}".format(self.episode))
	
	def on_done(self):
		raise NotImplementedError()

	def train_step(self, i):
		raise NotImplementedError()
