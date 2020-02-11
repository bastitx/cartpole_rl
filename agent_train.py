import gym
from ddpg_agent import DDPGAgent
from inverted_pendulum import InvertedPendulumEnv
from model import ActorModel, CriticModel
import torch
import sys
import argparse

def main():
    global env, agent, args, episode

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--max_iterations', default=100000, type=int)
    parser.add_argument('--warmup', default=128, type=int)
    parser.add_argument('--max_episode_length', default=500, type=int)
    parser.add_argument('--resume_episode', default=0, type=int)
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--epsilon_decay', default=0.9999, type=float)
    parser.add_argument('--epsilon_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lr_actor', default=0.0001, type=float)
    parser.add_argument('--lr_critic', default=0.001, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)

    args = parser.parse_args()

    env = InvertedPendulumEnv()

    env.seed(0)
    agent = DDPGAgent(env.observation_space, env.action_space, env.parameter_space, ActorModel, CriticModel, args.gamma, args.epsilon, args.epsilon_min,
                    args.epsilon_decay, args.lr_actor, args.lr_critic, args.tau, args.batch_size, args.memory_size)
    if args.resume_episode > 0:
        try:
            agent.load_weights('models', args.resume_episode, args.mode is not 'train')
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
            print("{}/{}: episode: {}, avg.reward: {:.3f}, length: {}, epsilon: {:.3f}".format(i,args.max_iterations, episode, episode_reward/episode_length, episode_length, agent.epsilon))
            env.close()
            env = InvertedPendulumEnv()
            env.seed(0)
            state = env.reset()
            episode_reward = 0
            last_episode_start = i
        if args.mode is not 'train':
            env.render()
        action = agent.act(state, env.params)
        next_state, reward, done, _ = env.step(action)
        #print(state[1], action[0], reward)
        episode_reward += reward
        agent.remember([state], [env.params], [action], [next_state], [reward], [done])
        if args.mode is 'train' and i >= args.warmup:
            agent.update()
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
            agent.save_model('models', episode)
            print("Saved model")
    sys.exit(0)