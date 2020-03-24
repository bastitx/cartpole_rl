import gym
from agents.ddpg_agent import DDPGAgent
from sim.cartpole import CartPoleEnv
from model import ActorModel, CriticModel
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

    env = CartPoleEnv(args.swingup, args.randomize)

    writer = SummaryWriter()

    state_shape = np.array(env.observation_space.shape)
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
            writer.add_scalar('EpisodeReward/total', episode_reward, episode)
            writer.add_scalar('EpisodeReward/avg', episode_reward/episode_length, episode)
            writer.add_scalar('EpisodeLength', episode_length, episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Memory', len(agent.memory.memory), episode)
            writer.flush()
            if args.randomize:
                env.randomize_params()
            state = env.reset()
            episode_reward = 0
            last_episode_start = i

        if args.mode != 'train':
            env.render()
        
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.remember([state], [action], [next_state], [reward], [done])

        if args.mode == 'train' and i >= args.warmup:
            agent.update()

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
