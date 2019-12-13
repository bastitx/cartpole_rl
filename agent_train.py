import gym
from ddpg_agent import DDPGAgent
from inverted_pendulum import InvertedPendulumEnv
from model import ActorModel, CriticModel
import torch
import sys

def main():
    global env, agent, epoch, train
    env = InvertedPendulumEnv()
    train = True
    max_iterations = 100000
    warmup = 512
    max_epoch_length = 200
    epoch = 0

    env.seed(0)
    agent = DDPGAgent(env.observation_space, env.action_space, ActorModel, CriticModel, epsilon=1.0)
    try:
        agent.load_weights('models', epoch, not train)
        print("loaded weights")
    except Exception:
        print("Couldn't load weights!")
    
    epoch_reward = 0
    last_epoch_start = -1
    done = True

    for i in range(max_iterations):
        if done:
            epoch += 1
            epoch_length = i - last_epoch_start
            print("{}/{}: epoch: {}, avg.reward: {:.3f}, length: {}, epsilon: {:.3f}".format(i,max_iterations, epoch, epoch_reward/epoch_length, epoch_length, agent.epsilon))
            state = env.reset()
            epoch_reward = 0
            last_epoch_start = i
        if not train:
            env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        #print(state[1], action[0], reward)
        epoch_reward += reward
        agent.remember([state], [action], [next_state], [reward], [done])
        if train and i >= warmup:
            agent.update()
        state = next_state
        if i - last_epoch_start >= max_epoch_length:
            done = True
        
    env.close()
    if train:
        agent.save_model('models', epoch)
        print("Saved model")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        env.close()
        if train:
            agent.save_model('models', epoch)
            print("Saved model")
    sys.exit(0)