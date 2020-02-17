import gym
from osi import OSI
from inverted_pendulum import InvertedPendulumEnv
from model import ActorModel, CriticModel
import torch
import sys
import argparse
from ddpg_agent import DDPGAgent
from collections import deque


def main():
	global env, osi, args, episode

	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='train', type=str)
	parser.add_argument('--max-iterations', default=100000, type=int)
	parser.add_argument('--warmup', default=128, type=int)
	parser.add_argument('--history-length', default=5, type=int)
	parser.add_argument('--episode-length', default=500, type=int)
	parser.add_argument('--resume-episode', default=0, type=int)
	parser.add_argument('--lr', default=0.0001, type=float)
	parser.add_argument('--batch-size', default=64, type=int)
	parser.add_argument('--memory-size', default=10000, type=int)

	args = parser.parse_args()

	env = InvertedPendulumEnv()
	env.seed(0)

	osi = OSI(2*args.history_length, len(env.params),
				args.lr, args.batch_size, args.memory_size)

	agent = DDPGAgent(env.observation_space, env.action_space,
						env.parameter_space, ActorModel, CriticModel)
	agent.load_weights('models', 2051, False)

	if args.resume_episode > 0:
		try:
			osi.load_weights('models', args.resume_episode,
								args.mode is not 'train')
			print("loaded weights")
		except Exception:
			print("Couldn't load weights!")
			
	episode = args.resume_episode
	last_episode_start = -1
	history_queue = []
	done = True

	for i in range(args.max_iterations):
		if done:
			episode += 1
			history_queue = deque()
			state = env.reset()
			last_episode_start = i

		if len(history_queue) == args.history_length:
			mu_ = osi.predict(history_queue_t[0], history_queue_t[1])
			action = agent.act(history_queue_t[0], mu_)
		else:
			action = agent.act(state, env.params)

		next_state, _, _, _ = env.step(action)
		history_queue.append([next_state, action])

		if len(history_queue) == args.history_length+1:
			history_queue.pop(0)
			history_queue_t = list(zip(*history_queue))
			osi.remember(history_queue_t[0], history_queue_t[1], [env.params])
			if args.mode is 'train' and i >= args.warmup:
				osi.update()

		state = next_state
		if i - last_episode_start >= args.max_episode_length:
			done = True

	env.close()
	if args.mode is 'train':
		agent.save_model('models', episode)
		print("Saved model")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        env.close()
        if args.mode is 'train':
            osi.save_model('models', episode)
            print("Saved model")
    sys.exit(0)
