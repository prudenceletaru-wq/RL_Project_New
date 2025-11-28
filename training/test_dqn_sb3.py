from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from hospital_env import HospitalEnv
import numpy as np

# --- Wrap environment ---
def make_env():
    env = HospitalEnv()
    env = Monitor(env)
    return env

eval_env = DummyVecEnv([make_env])

# --- Load trained model ---
model = DQN.load("../models/dqn_hospital_sb3", env=eval_env)

# --- Evaluate over multiple episodes ---
num_episodes = 10
episode_rewards = []

for ep in range(num_episodes):
    obs = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_array, info = eval_env.step(action)
        done = done_array[0]   # VecEnv returns arrays
        total_reward += reward[0]  # VecEnv returns arrays

    episode_rewards.append(total_reward)

print(f"Rewards over {num_episodes} episodes: {episode_rewards}")
print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
