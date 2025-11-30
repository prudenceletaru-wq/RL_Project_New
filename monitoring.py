import json
import time
import numpy as np
from collections import deque, Counter
from scipy.stats import ks_2samp
import logging
import os

# -------------------------------
# Setup logging
# -------------------------------
logging.basicConfig(
    filename="monitoring.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# Load training metrics
# -------------------------------
METRICS_FILE = "training_metrics.json"
if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r") as f:
        training_metrics = json.load(f)
    train_action_dist = np.array(training_metrics["train_action_dist"])
    train_wait_red = np.array(training_metrics["train_wait_red"])
    train_wait_yellow = np.array(training_metrics["train_wait_yellow"])
else:
    train_action_dist = np.array([0.5, 0.5])
    train_wait_red = np.random.normal(10, 3, 500)
    train_wait_yellow = np.random.normal(20, 5, 500)

# -------------------------------
# Monitoring buffers
# -------------------------------
recent_rewards = deque(maxlen=200)
recent_actions = deque(maxlen=200)
recent_wait_red = deque(maxlen=200)
recent_wait_yellow = deque(maxlen=200)

# -------------------------------
# Monitoring functions
# -------------------------------
def track_reward(r):
    recent_rewards.append(r)

def track_action(a):
    recent_actions.append(a)
    count = Counter(recent_actions)
    total = sum(count.values())
    dist = {"red": count.get(0, 0)/total if total else 0,
            "yellow": count.get(1, 0)/total if total else 0}
    # Model drift check
    drift = np.linalg.norm(np.array([dist["red"], dist["yellow"]]) - train_action_dist)
    if drift > 0.25:
        logging.warning("MODEL DRIFT DETECTED")
    return dist

def track_wait(cat, wt):
    if cat == "red":
        recent_wait_red.append(wt)
        if len(recent_wait_red) > 30:
            check_wait_drift("red")
    else:
        recent_wait_yellow.append(wt)
        if len(recent_wait_yellow) > 30:
            check_wait_drift("yellow")

def check_wait_drift(cat):
    if cat == "red":
        stat, p = ks_2samp(train_wait_red, list(recent_wait_red))
    else:
        stat, p = ks_2samp(train_wait_yellow, list(recent_wait_yellow))
    if p < 0.05:
        logging.warning(f"DATA DRIFT DETECTED in {cat.upper()} wait times (p={p:.4f})")

# -------------------------------
# Monitor API logs in real time
# -------------------------------
LOG_FILE = "api_logs.json"
processed_lines = 0

print("[INFO] Continuous monitoring started. Press Ctrl+C to stop.")

while True:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        new_lines = lines[processed_lines:]
        for line in new_lines:
            try:
                entry = json.loads(line)
                action = 0 if entry["action"] == "RED" else 1
                reward = entry["reward"]
                wait_time = entry["wait_time"]
                cat = "red" if action == 0 else "yellow"

                track_action(action)
                track_wait(cat, wait_time)
                track_reward(reward)
            except Exception as e:
                logging.error(f"Error processing log line: {e}")

        processed_lines += len(new_lines)

        # Optional: print periodic summary
        if len(recent_rewards) > 0 and processed_lines % 20 == 0:
            print(f"[SUMMARY] Mean reward: {np.mean(recent_rewards):.2f}, "
                  f"Action distribution: RED {recent_actions.count(0)}, YELLOW {recent_actions.count(1)}, "
                  f"Avg waits: RED {np.mean(recent_wait_red):.2f}, YELLOW {np.mean(recent_wait_yellow):.2f}")

    time.sleep(2)  # check every 2 seconds
