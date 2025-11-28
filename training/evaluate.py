import sys
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add environment path
sys.path.append("../env")  
from hospital_env import HospitalEnv

# -------------------------------
# Create wrapped evaluation environment
# -------------------------------
def make_env():
    env = HospitalEnv()
    env = Monitor(env)  # SB3 logging wrapper
    return env

eval_env = DummyVecEnv([make_env])

# -------------------------------
# Load trained DQN model
# -------------------------------
model = DQN.load("../models/dqn_hospital_sb3", env=eval_env)

# -------------------------------
# Evaluation parameters
# -------------------------------
n_episodes = 10  # number of episodes to evaluate
threshold_times = {"red": 10, "yellow": 30, "green": 60}

# Storage for metrics
rewards_per_episode = []
red_waits, yellow_waits, green_waits = [], [], []
queue_lengths = {"red": [], "yellow": [], "green": []}
green_served_count = 0
total_served_count = 0

# -------------------------------
# Evaluate model
# -------------------------------
for ep in range(n_episodes):
    obs = eval_env.reset()  # only obs returned
    done = False
    episode_reward = 0

    while not done:
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]

        # Access raw environment
        env = eval_env.envs[0].unwrapped

        # Record last served wait times
        if env.last_served_wait_times["red"] > 0:
            red_waits.append(env.last_served_wait_times["red"])
            total_served_count += 1
        if env.last_served_wait_times["yellow"] > 0:
            yellow_waits.append(env.last_served_wait_times["yellow"])
            total_served_count += 1
        if env.last_served_wait_times["green"] > 0:
            green_waits.append(env.last_served_wait_times["green"])
            green_served_count += 1
            total_served_count += 1

        # Record queue lengths
        queue_lengths["red"].append(len(env.red_queue))
        queue_lengths["yellow"].append(len(env.yellow_queue))
        queue_lengths["green"].append(len(env.green_queue))

    rewards_per_episode.append(episode_reward)

# -------------------------------
# Compute metrics
# -------------------------------
avg_reward = np.mean(rewards_per_episode)
avg_wait_red = np.mean(red_waits) if red_waits else 0
avg_wait_yellow = np.mean(yellow_waits) if yellow_waits else 0
avg_wait_green = np.mean(green_waits) if green_waits else 0

pct_red_within = 100 * sum(w <= threshold_times["red"] for w in red_waits) / len(red_waits) if red_waits else 0
pct_yellow_within = 100 * sum(w <= threshold_times["yellow"] for w in yellow_waits) / len(yellow_waits) if yellow_waits else 0
pct_green_within = 100 * sum(w <= threshold_times["green"] for w in green_waits) / len(green_waits) if green_waits else 0

queue_stats = {cat: {"avg": np.mean(qs), "max": np.max(qs)} for cat, qs in queue_lengths.items()}

# Fairness: GREEN patients served / total served
fairness = green_served_count / total_served_count if total_served_count > 0 else 0

# -------------------------------
# Print results
# -------------------------------
print(f"Average reward per episode: {avg_reward:.2f}")
print(f"Average wait times (Red, Yellow, Green): {avg_wait_red:.2f}, {avg_wait_yellow:.2f}, {avg_wait_green:.2f}")
print(f"Percentage served within thresholds (Red, Yellow, Green): {pct_red_within:.2f}%, {pct_yellow_within:.2f}%, {pct_green_within:.2f}%")
print("Queue stats (average and max lengths):")
for cat, stats in queue_stats.items():
    print(f"  {cat.capitalize()}: avg={stats['avg']:.2f}, max={stats['max']:.2f}")
print(f"Fairness (GREEN served / total served): {fairness:.2f}")
