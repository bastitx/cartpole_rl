import gym
from sim.cartpole_dc import CartPoleEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

env = CartPoleEnv(swingup=False, randomize=False)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
dones = False
for i in range(1000):
    action, _states = model.predict(obs)
    if dones:
        obs = env.reset()
        dones = False
    else:
        obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
