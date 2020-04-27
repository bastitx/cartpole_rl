import gym
from agents.ppo_agent import PPOAgent
from sim.cartpole_dc import CartPoleEnv
from model import ActorCriticModel
import sys
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
	global env, agent, args, episode

	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='train', type=str)
	parser.add_argument('--max-iterations', default=100000, type=int)
	parser.add_argument('--max-episode-length', default=500, type=int)
	parser.add_argument('--resume-episode', default=0, type=int)
	parser.add_argument('--gamma', default=0.99, type=float)
	parser.add_argument('--lr', default=0.0001, type=float)
	parser.add_argument('--batch-size', default=256, type=int)
	parser.add_argument('--memory-size', default=4096, type=int)
	parser.add_argument('--epochs', default=16, type=int)
	parser.add_argument('--init-variance', default=0.25, type=float)
	parser.add_argument('--randomize', dest='randomize', action='store_true')
	parser.set_defaults(randomize=False)
	parser.add_argument('--render', dest='render', action='store_true')
	parser.set_defaults(render=False)
	parser.add_argument('--no-swingup', dest='swingup', action='store_false')
	parser.set_defaults(swingup=True)

	args = parser.parse_args()

	env = CartPoleEnv(args.swingup, observe_params=args.randomize)

	writer = SummaryWriter()
	agent = PPOAgent((16,), env.action_space.shape, ActorCriticModel, args.gamma, 
					args.lr, args.batch_size, args.epochs, args.memory_size, args.init_variance)

	if args.resume_episode > 0:
		try:
			agent.load_weights('models', args.resume_episode)
			print("loaded weights")
			episode = args.resume_episode
		except Exception as e:
			print("Couldn't load weights! {}".format(e))
			episode = 0
	else:
		episode = 0

	episode_reward = 0
	last_episode_start = -1
	done = True

	for i in range(args.max_iterations):
		if done:
			episode += 1
			episode_length = i - last_episode_start
			writer.add_scalar('EpisodeReward/total', episode_reward, episode)
			writer.add_scalar('EpisodeReward/avg', episode_reward/episode_length, episode)
			writer.add_scalar('EpisodeLength', episode_length, episode)
			writer.add_scalar('Variance', agent.policy.action_var, episode)
			writer.add_scalar('Memory', len(agent.memory.memory), episode)
			writer.flush()
			comp_state = torch.zeros((4,4))
			state = env.reset(variance=0.05)
			state = state.detach()
			#writer.add_scalar('StateValue', agent.policy_old.critic(state), episode)
			if args.randomize:
				env.randomize_params()
			episode_reward = 0
			last_episode_start = i

		if args.render:
			env.render()
		
		comp_state = torch.cat((state[0,None], comp_state[:-1]))

		action, logprob = agent.act(comp_state.flatten().unsqueeze(0))
		next_state, reward, done, _ = env.step(action)
		next_state = next_state.detach()
		episode_reward += reward
		agent.remember(comp_state.flatten().unsqueeze(0), action, logprob, next_state, reward.detach(), done.detach())

		if args.mode == 'train' and i % args.memory_size == 0 and i > 0:
			avg_loss = agent.update()
			writer.add_scalar('Loss/Avg', avg_loss, i)
			agent.memory.clear()
			if args.randomize:
				done = True

		state = next_state

		if i - last_episode_start >= args.max_episode_length:
			done = True

	env.close()
	if args.mode == 'train':
		agent.save_model('models', episode)
		print("Saved model: {}".format(episode))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        env.close()
        if args.mode == 'train':
            agent.save_model('models', episode)
            print("Saved model: {}".format(episode))
    sys.exit(0)
