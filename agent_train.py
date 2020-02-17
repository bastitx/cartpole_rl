import gym
from ddpg_agent import DDPGAgent
from inverted_pendulum import InvertedPendulumEnv
from model import ActorModel, CriticModel
import torch
import sys
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def main():
    global env, agent, args, episode

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--max-iterations', default=100000, type=int)
    parser.add_argument('--warmup', default=128, type=int)
    parser.add_argument('--max_episode-length', default=500, type=int)
    parser.add_argument('--resume-episode', default=0, type=int)
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--epsilon-decay', default=0.9999, type=float)
    parser.add_argument('--epsilon-min', default=0.1, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lr-actor', default=0.0001, type=float)
    parser.add_argument('--lr-critic', default=0.001, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--memory-size', default=10000, type=int)
    parser.add_argument('--randomize', dest='randomize', action='store_true')
    parser.set_defaults(randomize=False)
    parser.add_argument('--no-swingup', dest='swingup', action='store_false')
    parser.set_defaults(swingup=True)

    args = parser.parse_args()

    env = InvertedPendulumEnv(args.swingup)
    env.seed(0)

    writer = SummaryWriter()

    state_shape = np.array(env.observation_space.shape)
    state_shape += np.array(env.params.shape)
    agent = DDPGAgent(tuple(state_shape), env.action_space.shape, ActorModel, CriticModel, args.gamma, args.epsilon, args.epsilon_min,
                      args.epsilon_decay, args.lr_actor, args.lr_critic, args.tau, args.batch_size, args.memory_size)
    #writer.add_graph(agent.actor)
    #writer.add_graph(agent.critic)
    
    if args.resume_episode > 0:
        try:
            agent.load_weights('models', args.resume_episode, args.mode == 'train')
            print("loaded weights")
        except Exception:
            print("Couldn't load weights!")

    episode = args.resume_episode
    episode_reward = 0
    last_episode_start = -1
    done = True

    for i in range(args.max_iterations):
        if done:
            episode += 1
            episode_length = i - last_episode_start
            #print("{}/{}: episode: {}, avg.reward: {:.3f}, length: {}, epsilon: {:.3f}".format(
            #    i, args.max_iterations, episode, episode_reward/episode_length,
            #    episode_length, agent.epsilon))
            writer.add_scalar('EpisodeReward/total', episode_reward, episode)
            writer.add_scalar('EpisodeReward/avg', episode_reward/episode_length, episode)
            writer.add_scalar('EpisodeLength', episode_length, episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Memory', len(agent.memory.memory), episode)
            writer.flush()
            if args.randomize:
                env.close()
                env.reinit()
            state = env.reset()
            comp_state = np.concatenate((state, env.params))
            episode_reward = 0
            last_episode_start = i

        if args.mode is not 'train':
            env.render()
        
        action = agent.act(comp_state)
        next_state, reward, done, _ = env.step(action)
        #print(state[1], action[0], reward)
        episode_reward += reward
        comp_next_state = np.concatenate((next_state, env.params))
        agent.remember([comp_state], [action], [comp_next_state], [reward], [done])

        if args.mode is 'train' and i >= args.warmup:
            agent.update()

        state = next_state
        comp_state = comp_next_state

        if i - last_episode_start >= args.max_episode_length:
            done = True

    env.close()
    if args.mode is 'train':
        agent.save_model('models', episode)
        print("Saved model: {}".format(episode))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        env.close()
        if args.mode is 'train':
            agent.save_model('models', episode)
            print("Saved model: {}".format(episode))
    sys.exit(0)
