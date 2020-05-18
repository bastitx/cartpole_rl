import gym
from agents.osi import OSI
from sim.cartpole_dc import CartPoleEnv
from model import ActorCriticModel
import torch
import sys
import argparse
from agents.ppo_agent import PPOAgent
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def main():
	global env, osi, args, episode

	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='train', type=str)
	parser.add_argument('--max-iterations', default=100000, type=int)
	parser.add_argument('--steps-per-param', default=500, type=int)
	parser.add_argument('--history-length', default=5, type=int)
	parser.add_argument('--max-episode-length', default=300, type=int)
	parser.add_argument('--resume-episode', default=0, type=int)
	parser.add_argument('--lr', default=0.0001, type=float)
	parser.add_argument('--batch-size', default=64, type=int)
	parser.add_argument('--epochs', default=200, type=int)
	parser.add_argument('--training-samples', default=16384, type=int)
	parser.add_argument('--memory-size', default=100000, type=int)

	args = parser.parse_args()

	env = CartPoleEnv(swingup=False, observe_params=False)
	env.randomize_params()

	writer = SummaryWriter()

	osi = OSI(5*args.history_length, len(env.params), args.lr, args.batch_size, args.memory_size, args.epochs)

	agent = PPOAgent(env.param_observation_space.shape, env.action_space.shape, ActorCriticModel)
	agent.load_weights('models', 3409, False)

	if args.resume_episode > 0:
		try:
			osi.load_weights('models', args.resume_episode, args.mode != 'train')
			print("loaded weights")
		except Exception as e:
			print("Couldn't load weights! {}".format(e))
			
	episode = args.resume_episode
	last_episode_start = -1
	last_param_change = 0
	history_queue = deque()
	done = True
	log_counter = 0
	mem_counter = 0

	for i in range(args.max_iterations):
		if done:
			episode += 1
			episode_length = i - last_episode_start
			writer.add_scalar('EpisodeLength', episode_length, episode)
			writer.add_scalar('Memory', len(osi.memory), episode)
			writer.flush()
			history_queue = deque()
			osi_state = None
			state = env.reset()
			last_episode_start = i
		
		if args.mode != 'train':
			env.render()

		if osi_state is not None and episode > 1:
			mu = osi.predict(osi_state)
			action, _ = agent.act(np.concatenate((state, mu)))
		else:
			action, _ = agent.act(np.concatenate((state, env.params)))

		state, _, done, _ = env.step(action) # add action noise?
		history_queue.append([state, action])

		if len(history_queue) == args.history_length+1:
			history_queue.popleft()
			history_queue_t = list(zip(*history_queue))
			osi_state = np.concatenate((*history_queue_t[0], *history_queue_t[1]))
			osi.remember(osi_state, [env.params])
			mem_counter += 1

		if i - last_episode_start >= args.max_episode_length:
			done = True

		if i - last_param_change == args.steps_per_param and episode > 1:
			last_param_change = i
			env.randomize_params()
			done = True
		
		if args.mode == 'train' and mem_counter % args.training_samples == 0 and mem_counter > 0:
			losses = osi.update()
			for l in losses:
				writer.add_scalar('Loss', l, log_counter)
				log_counter += 1
			writer.flush()

	env.close()
	if args.mode == 'train':
		osi.save_model('models', episode)
		print("Saved model {}".format(episode))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        env.close()
        if args.mode == 'train':
            osi.save_model('models', episode)
            print("Saved model {}".format(episode))
    sys.exit(0)
