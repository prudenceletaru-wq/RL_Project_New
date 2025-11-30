import sys
import numpy as np
import json
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
n_episodes = 40
threshold_times = {"red": 15, "yellow": 30}

# Storage for metrics
rewards_per_episode = []
red_waits, yellow_waits = [], []
queue_lengths = {"red": [], "yellow": []}
total_actions_red, total_actions_yellow = 0, 0

# -------------------------------
# Model Evaluation
# -------------------------------
for ep in range(n_episodes):
    obs = eval_env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)

        # Count actions (0 = Red = urgent, 1 = Yellow = non-urgent)
        if action[0] == 0:
            total_actions_red += 1
        else:
            total_actions_yellow += 1

        obs, reward, done, info = eval_env.step(action)
        episode_reward += float(reward[0])

        env = eval_env.envs[0].unwrapped

        # Recording wait times
        if env.last_served_wait_times["red"] > 0:
            red_waits.append(float(env.last_served_wait_times["red"]))
        if env.last_served_wait_times["yellow"] > 0:
            yellow_waits.append(float(env.last_served_wait_times["yellow"]))

        # Recording queue lengths
        queue_lengths["red"].append(len(env.red_queue))
        queue_lengths["yellow"].append(len(env.yellow_queue))

    rewards_per_episode.append(episode_reward)

# -------------------------------
# Computing metrics
# -------------------------------
avg_reward = float(np.mean(rewards_per_episode))
avg_wait_red = float(np.mean(red_waits)) if red_waits else 0
avg_wait_yellow = float(np.mean(yellow_waits)) if yellow_waits else 0

pct_red_within = float(100 * sum(w <= threshold_times["red"] for w in red_waits) / len(red_waits)) if red_waits else 0
pct_yellow_within = float(100 * sum(w <= threshold_times["yellow"] for w in yellow_waits) / len(yellow_waits)) if yellow_waits else 0

queue_stats = {cat: {"avg": float(np.mean(qs)), "max": int(np.max(qs))} for cat, qs in queue_lengths.items()}

# -------------------------------
# ACTION DISTRIBUTION
# -------------------------------
total_actions = total_actions_red + total_actions_yellow
red_pct = float(total_actions_red / total_actions) if total_actions > 0 else 0
yellow_pct = float(total_actions_yellow / total_actions) if total_actions > 0 else 0

print("\n=== ACTION DISTRIBUTION (TRAINING BEHAVIOR) ===")
print("Red = urgent patients  | Yellow = non-urgent patients")
print(f"Red actions: {total_actions_red} ({red_pct:.3f})")
print(f"Yellow actions: {total_actions_yellow} ({yellow_pct:.3f})")

print("\n=== PERFORMANCE METRICS ===")
print(f"Average reward per episode: {avg_reward:.2f}")
print(f"Average wait times (Red, Yellow): {avg_wait_red:.2f}, {avg_wait_yellow:.2f}")
print(f"Percentage served within thresholds (Red, Yellow): {pct_red_within:.2f}%, {pct_yellow_within:.2f}%")
print("Queue stats (average and max lengths):")
for cat, stats in queue_stats.items():
    print(f"  {cat.capitalize()}: avg={stats['avg']:.2f}, max={stats['max']}")

# -------------------------------
# Saving training metrics to JSON
# -------------------------------
training_metrics = {
    "train_action_dist": [red_pct, yellow_pct],
    "train_wait_red": red_waits,
    "train_wait_yellow": yellow_waits,
    "avg_reward_per_episode": avg_reward,
    "pct_within_threshold_red": pct_red_within,
    "pct_within_threshold_yellow": pct_yellow_within,
    "queue_red_avg": queue_stats["red"]["avg"],
    "queue_red_max": queue_stats["red"]["max"],
    "queue_yellow_avg": queue_stats["yellow"]["avg"],
    "queue_yellow_max": queue_stats["yellow"]["max"]
}

with open("../training_metrics.json", "w") as f:
    json.dump(training_metrics, f, indent=4)

print("\n[INFO] Training metrics saved to 'training_metrics.json'")
