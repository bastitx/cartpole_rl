import sim.inverted_pendulum as inverted_pendulum
import agents.random_agent as random_agent
import gym
import numpy as np

agent = random_agent.RandomAgent(gym.spaces.Box(np.array([-1]), np.array([1])))
env = inverted_pendulum.InvertedPendulumEnv()
for _ in range(2000):
    state = env.reset()
    env.render()
    for i in range(200):
        state = env.step(agent.act(state))
        env.render()
    env.close()
