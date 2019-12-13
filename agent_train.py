import gym
from ddpg_agent import DDPGAgent
from inverted_pendulum import InvertedPendulumEnv
from model import ActorModel, CriticModel
import torch
import sys

def main():
    global env, agent, epoch
    env = InvertedPendulumEnv()

    env.seed(0)
    agent = DDPGAgent(env.observation_space, env.action_space, ActorModel, CriticModel, epsilon=0.1)
    epoch = 2299
    try:
        agent.load_weights('models', epoch)
        print("loaded weights")
    except Exception:
        print("Couldn't load weights!")

    max_iterations = 100000
    warmup = 20000
    max_epoch_length = 400
    train = True
    
    epoch_reward = 0
    last_epoch_start = -1
    done = True

    for i in range(max_iterations):
        if done:
            epoch += 1
            epoch_length = i - last_epoch_start
            print("{}/{}: avg.reward: {}, length: {}, epsilon: {}".format(i,max_iterations, epoch_reward/epoch_length, epoch_length, agent.epsilon))
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
    agent.save_model('models', epoch, epoch)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        env.close()
        agent.save_model('models', epoch, epoch)
        print("Saved model")
    sys.exit(0)