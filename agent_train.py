import gym
from ddpg_agent import DDPGAgent
from inverted_pendulum import InvertedPendulumEnv
from model import ActorModel, CriticModel
import torch

def main():
    env = InvertedPendulumEnv()

    env.seed(0)
    agent = DDPGAgent(env.observation_space, env.action_space, ActorModel, CriticModel)
    episode_count = 1000
    max_episode_length = 1000

    for i in range(episode_count):
        state = env.reset()
        episode_reward = 0
        for j in range(max_episode_length):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #print(state[1], action[0], reward)
            episode_reward += reward
            agent.remember([state], [action], [next_state], [reward], [done])
            agent.update()
            state = next_state
            if done:
                break
        print("{}: reward: {}, length: {}, epsilon: {}".format(i, episode_reward, j, agent.epsilon))
        
    env.close()


if __name__ == '__main__':
    main()