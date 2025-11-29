import sys
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Adding environment path
sys.path.append("../env")  
from hospital_env import HospitalEnv

# -------------------------------
# Creating wrapped evaluation environment
# -------------------------------
def make_env():
    env = HospitalEnv()
    env = Monitor(env)  # SB3 logging wrapper
    return env

eval_env = DummyVecEnv([make_env])

# -------------------------------
# Loading trained DQN model
# -------------------------------
model = DQN.load("../models/dqn_hospital_sb3", env=eval_env)

# -------------------------------
# Evaluation parameters
# -------------------------------
n_episodes = 40  # number of episodes to evaluate
threshold_times = {"red": 15, "yellow": 30}

# Storage for metrics
rewards_per_episode = []
red_waits, yellow_waits = [], []
queue_lengths = {"red": [], "yellow": []}
total_served_count = 0

# -------------------------------
# Model Evaluation
# -------------------------------
for ep in range(n_episodes):
    obs = eval_env.reset()  # only obs returned
    done = False
    episode_reward = 0

    while not done:
        # Predicting action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]

        # Accessing raw environment
        env = eval_env.envs[0].unwrapped

        # Recording last served wait times
        if env.last_served_wait_times["red"] > 0:
            red_waits.append(env.last_served_wait_times["red"])
            total_served_count += 1
        if env.last_served_wait_times["yellow"] > 0:
            yellow_waits.append(env.last_served_wait_times["yellow"])
            total_served_count += 1

        # Recording queue lengths
        queue_lengths["red"].append(len(env.red_queue))
        queue_lengths["yellow"].append(len(env.yellow_queue))

    rewards_per_episode.append(episode_reward)

# -------------------------------
# Computing metrics
# -------------------------------
avg_reward = np.mean(rewards_per_episode)
avg_wait_red = np.mean(red_waits) if red_waits else 0
avg_wait_yellow = np.mean(yellow_waits) if yellow_waits else 0

pct_red_within = 100 * sum(w <= threshold_times["red"] for w in red_waits) / len(red_waits) if red_waits else 0
pct_yellow_within = 100 * sum(w <= threshold_times["yellow"] for w in yellow_waits) / len(yellow_waits) if yellow_waits else 0

queue_stats = {cat: {"avg": np.mean(qs), "max": np.max(qs)} for cat, qs in queue_lengths.items()}

# -------------------------------
# Printing results
# -------------------------------
print(f"Average reward per episode: {avg_reward:.2f}")
print(f"Average wait times (Red, Yellow): {avg_wait_red:.2f}, {avg_wait_yellow:.2f}")
print(f"Percentage served within thresholds (Red, Yellow): {pct_red_within:.2f}%, {pct_yellow_within:.2f}%")
print("Queue stats (average and max lengths):")
for cat, stats in queue_stats.items():
    print(f"  {cat.capitalize()}: avg={stats['avg']:.2f}, max={stats['max']:.2f}")