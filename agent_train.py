import gym
from ddpg_agent import DDPGAgent
from inverted_pendulum import InvertedPendulumEnv
from model import ActorModel, CriticModel
import torch
import sys

def main():
    global env, agent, episode, train
    env = InvertedPendulumEnv()
    train = True
    max_iterations = 100000
    warmup = 128
    max_episode_length = 100
    episode = 0

    env.seed(0)
    agent = DDPGAgent(env.observation_space, env.action_space, ActorModel, CriticModel, epsilon=1.0)
    try:
        agent.load_weights('models', episode, not train)
        print("loaded weights")
    except Exception:
        print("Couldn't load weights!")
    
    episode_reward = 0
    last_episode_start = -1
    done = True

    for i in range(max_iterations):
        if done:
            episode += 1
            episode_length = i - last_episode_start
            print("{}/{}: epoch: {}, avg.reward: {:.3f}, length: {}, epsilon: {:.3f}".format(i,max_iterations, episode, episode_reward/episode_length, episode_length, agent.epsilon))
            state = env.reset()
            episode_reward = 0
            last_episode_start = i
        if not train:
            env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        #print(state[1], action[0], reward)
        episode_reward += reward
        agent.remember([state], [action], [next_state], [reward], [done])
        if train and i >= warmup:
            agent.update()
        state = next_state
        if i - last_episode_start >= max_episode_length:
            done = True
        
    env.close()
    if train:
        agent.save_model('models', episode)
        print("Saved model")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        env.close()
        if train:
            agent.save_model('models', episode)
            print("Saved model")
    sys.exit(0)