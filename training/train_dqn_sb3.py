import warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained since")

import sys
sys.path.append("../env")  # to read hospital_env.py
from hospital_env import HospitalEnv

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ------------------------------
# Set seeds for reproducibility
# ------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Create wrapped environment ---
def make_env():
    env = HospitalEnv()
    env.reset(seed=SEED)   # seed the environment here
    env = Monitor(env)      # important for SB3 logging
    return env

env = DummyVecEnv([make_env])

# --- Create DQN agent ---
model = DQN(
    "MlpPolicy",       # Fully connected NN
    env,
    learning_rate=5e-4,
    gamma=0.95,
    batch_size=64,
    buffer_size=50000,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    exploration_fraction=0.1,   # epsilon decay
    target_update_interval=1000,
    verbose=1,
    seed=SEED                  # seed SB3 agent
)

# --- Train agent ---
model.learn(total_timesteps=50000)

# --- Save trained model ---
model.save("../models/dqn_hospital_sb3")
print("Model saved to models/dqn_hospital_sb3.zip")

# --- Evaluation ---
eval_env = DummyVecEnv([make_env])
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")