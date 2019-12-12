import gym
from ddpg_agent import DDPGAgent
from inverted_pendulum import InvertedPendulumEnv
from model import ActorModel, CriticModel
import torch
import sys

def main():
    env = InvertedPendulumEnv()
    global agent

    env.seed(0)
    agent = DDPGAgent(env.observation_space, env.action_space, ActorModel, CriticModel)
    try:
        agent.load_weights('models')
        print("loaded weights")
    except Exception:
        print("Couldn't load weights!")
    episode_count = 500
    max_episode_length = 400

    for i in range(episode_count):
        state = env.reset()
        episode_reward = 0
        for j in range(max_episode_length):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #print(state[1], action[0], reward)
            episode_reward += reward
            agent.remember([state], [action], [next_state], [reward], [done])
            agent.update()
            state = next_state
            if done:
                break
        print("{}: avg.reward: {}, length: {}, epsilon: {}".format(i, episode_reward/(j+1), j+1, agent.epsilon))
        
    env.close()
    agent.save_model('models')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        agent.save_model('models')
        print("Saved model")
    sys.exit(0)