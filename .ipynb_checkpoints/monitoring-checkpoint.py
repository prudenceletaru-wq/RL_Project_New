import sys
import os
import json
import time
import logging
from collections import deque, Counter
import numpy as np
from scipy.stats import ks_2samp

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# -------------------------------
# Setup logging
# -------------------------------
logging.basicConfig(
    filename="monitoring.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# Global buffers for drift detection
# -------------------------------
recent_rewards = deque(maxlen=200)
recent_actions = deque(maxlen=200)
recent_wait_red = deque(maxlen=200)
recent_wait_yellow = deque(maxlen=200)

# -------------------------------
# Monitoring functions
# -------------------------------
def track_reward(reward):
    recent_rewards.append(reward)
    logging.info(f"Reward logged: {reward}")

def track_action(action):
    recent_actions.append(action)
    count = Counter(recent_actions)
    total = sum(count.values())
    dist = {
        "red": count.get(0, 0) / total if total else 0,
        "yellow": count.get(1, 0) / total if total else 0
    }
    logging.info(f"Action distribution: {dist}")

    # Model drift detection
    action_vector = np.array([dist["red"], dist["yellow"]])
    if "train_action_dist" in globals():
        drift = np.linalg.norm(action_vector - train_action_dist)
        if drift > 0.25:
            logging.warning("MODEL DRIFT DETECTED: Action distribution deviates from training!")

def track_wait_time(cat, wait_time):
    if cat == "red":
        recent_wait_red.append(wait_time)
        if len(recent_wait_red) > 30:
            _check_wait_time_drift("red")
    else:
        recent_wait_yellow.append(wait_time)
        if len(recent_wait_yellow) > 30:
            _check_wait_time_drift("yellow")

def _check_wait_time_drift(cat):
    if cat == "red":
        stat, p = ks_2samp(train_wait_red, list(recent_wait_red))
    else:
        stat, p = ks_2samp(train_wait_yellow, list(recent_wait_yellow))

    if p < 0.05:
        logging.warning(f"DATA DRIFT DETECTED in {cat.upper()} wait times (p={p:.4f})")

def log_decision(obs, action, reward, info={}):
    # Recursive conversion of NumPy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    obs = convert(obs)
    info = convert(info)

    data = {
        "timestamp": time.time(),
        "action": int(action),
        "reward": float(reward),
        "free_doctors": int(obs[0]),
        "longest_wait_red": float(obs[1]),
        "longest_wait_yellow": float(obs[2]),
        "red_queue_len": int(obs[3]),
        "yellow_queue_len": int(obs[4]),
        "doctor_busy_times": [float(obs[5]), float(obs[6]), float(obs[7])],
        "additional_info": info
    }
    logging.info("Decision: " + json.dumps(data))

# -------------------------------
# Main loop: run actual environment
# -------------------------------
if __name__ == "__main__":
    root_dir = "C:/Users/Prudence Letaru/Desktop/RL_Project_New"
    sys.path.append(root_dir)
    sys.path.append(os.path.join(root_dir, "env"))

    from hospital_env import HospitalEnv

    # Create evaluation environment
    def make_env():
        env = HospitalEnv()
        env = Monitor(env)
        return env

    eval_env = DummyVecEnv([make_env])

    # Load trained model
    model = DQN.load(os.path.join(root_dir, "models/dqn_hospital_sb3"), env=eval_env)

    # Load previous training metrics if available
    metrics_file = os.path.join(root_dir, "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            training_metrics = json.load(f)
        train_action_dist = np.array(training_metrics["train_action_dist"])
        train_wait_red = np.array(training_metrics["train_wait_red"])
        train_wait_yellow = np.array(training_metrics["train_wait_yellow"])
    else:
        # fallback: uniform distribution
        train_action_dist = np.array([0.5, 0.5])
        train_wait_red = np.random.normal(10, 3, 500)
        train_wait_yellow = np.random.normal(20, 5, 500)

    # Evaluation parameters
    n_episodes = 40
    threshold_times = {"red": 15, "yellow": 30}

    # Metrics storage
    rewards_per_episode = []
    red_waits, yellow_waits = [], []
    queue_lengths = {"red": [], "yellow": []}
    total_actions_red, total_actions_yellow = 0, 0

    # -------------------------------
    # Run episodes
    # -------------------------------
    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            if action[0] == 0:
                total_actions_red += 1
            else:
                total_actions_yellow += 1

            obs, reward, done, info = eval_env.step(action)
            episode_reward += float(reward[0])

            env = eval_env.envs[0].unwrapped

            # Track metrics
            track_reward(float(reward[0]))
            track_action(action[0])

            # Track wait times only if someone was served
            if env.last_served_wait_times["red"] > 0:
                red_waits.append(float(env.last_served_wait_times["red"]))
                track_wait_time("red", float(env.last_served_wait_times["red"]))
            if env.last_served_wait_times["yellow"] > 0:
                yellow_waits.append(float(env.last_served_wait_times["yellow"]))
                track_wait_time("yellow", float(env.last_served_wait_times["yellow"]))

            # Log decision (obs converted inside function)
            log_decision(obs[0], action[0], float(reward[0]), info)

            # Queue lengths
            queue_lengths["red"].append(len(env.red_queue))
            queue_lengths["yellow"].append(len(env.yellow_queue))

        rewards_per_episode.append(episode_reward)

    # -------------------------------
    # Compute summary metrics
    # -------------------------------
    avg_reward = float(np.mean(rewards_per_episode))
    avg_wait_red = float(np.mean(red_waits)) if red_waits else 0
    avg_wait_yellow = float(np.mean(yellow_waits)) if yellow_waits else 0
    pct_red_within = float(100 * sum(w <= threshold_times["red"] for w in red_waits) / len(red_waits)) if red_waits else 0
    pct_yellow_within = float(100 * sum(w <= threshold_times["yellow"] for w in yellow_waits) / len(yellow_waits)) if yellow_waits else 0
    queue_stats = {cat: {"avg": float(np.mean(qs)), "max": int(np.max(qs))} for cat, qs in queue_lengths.items()}

    # Action distribution
    total_actions = total_actions_red + total_actions_yellow
    red_pct = float(total_actions_red / total_actions) if total_actions > 0 else 0
    yellow_pct = float(total_actions_yellow / total_actions) if total_actions > 0 else 0

    # -------------------------------
    # Print summary
    # -------------------------------
    print("\n=== ACTION DISTRIBUTION ===")
    print(f"Red actions: {total_actions_red} ({red_pct:.3f})")
    print(f"Yellow actions: {total_actions_yellow} ({yellow_pct:.3f})")
    print("\n=== PERFORMANCE METRICS ===")
    print(f"Average reward per episode: {avg_reward:.2f}")
    print(f"Average wait times (Red, Yellow): {avg_wait_red:.2f}, {avg_wait_yellow:.2f}")
    print(f"Percentage served within thresholds (Red, Yellow): {pct_red_within:.2f}%, {pct_yellow_within:.2f}%")
    for cat, stats in queue_stats.items():
        print(f"{cat.capitalize()}: avg={stats['avg']:.2f}, max={stats['max']}")

    # -------------------------------
    # Save metrics
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

    with open(metrics_file, "w") as f:
        json.dump(training_metrics, f, indent=4)

    print("\n[INFO] Monitoring complete. Logs in 'monitoring.log'. Metrics saved in 'training_metrics.json'.")
