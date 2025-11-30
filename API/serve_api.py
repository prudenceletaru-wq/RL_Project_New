from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import os
import time
import sys

# -------------------------------
# Add project root to sys.path
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from env.hospital_env import HospitalEnv
from stable_baselines3 import DQN

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Hospital RL Agent API")

# -------------------------------
# Pydantic models for input
# -------------------------------
class State(BaseModel):
    free_doctors: int
    longest_wait_red: float
    longest_wait_yellow: float
    red_queue_length: int
    yellow_queue_length: int
    doctor1_busy_time: float
    doctor2_busy_time: float
    doctor3_busy_time: float

class RequestBody(BaseModel):
    state: State

# -------------------------------
# Load trained RL model
# -------------------------------
MODEL_PATH = os.path.join(ROOT_DIR, "models/dqn_hospital_sb3")
model = DQN.load(MODEL_PATH)

# -------------------------------
# Logging location for monitoring
# -------------------------------
LOG_FILE = os.path.join(ROOT_DIR, "api_logs.json")

# -------------------------------
# Convert API state to environment observation
# -------------------------------
def state_to_obs(state: State):
    return np.array([
        state.free_doctors,
        state.longest_wait_red,
        state.longest_wait_yellow,
        state.red_queue_length,
        state.yellow_queue_length,
        state.doctor1_busy_time,
        state.doctor2_busy_time,
        state.doctor3_busy_time
    ], dtype=np.float32)

# -------------------------------
# API endpoint
# -------------------------------
@app.post("/predict")
def predict(request: RequestBody):
    state = request.state

    # Compute free doctors from busy times
    doctor_busy_times = [
        state.doctor1_busy_time,
        state.doctor2_busy_time,
        state.doctor3_busy_time
    ]
    computed_free_doctors = sum(1 for t in doctor_busy_times if t == 0)

    # Validate input consistency
    if computed_free_doctors != state.free_doctors:
        return {
            "error": f"Inconsistent state: free_doctors={state.free_doctors} "
                     f"Does not match busy times (computed_free_doctors={computed_free_doctors})"
        }

    # Pre-check: Are queues empty?
    if state.red_queue_length == 0 and state.yellow_queue_length == 0:
        return {"message": "All queues are empty. No patients to attend."}

    # Pre-check: Are all doctors busy?
    if state.free_doctors == 0:
        return {"message": "All doctors are busy. Please wait."}

    # Convert state to observation for the model
    obs = state_to_obs(state)

    # Predict action
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    # Validate action against queue lengths
    if action == 0 and state.red_queue_length == 0:
        action = 1 if state.yellow_queue_length > 0 else None
    elif action == 1 and state.yellow_queue_length == 0:
        action = 0 if state.red_queue_length > 0 else None

    if action is None:
        return {"message": "No patients in the chosen queue."}

    # Compute reward and wait time using environment logic
    env = HospitalEnv()
    env.doctor_timers = np.array(doctor_busy_times, dtype=np.float32)
    env.red_queue = [state.longest_wait_red] * state.red_queue_length
    env.yellow_queue = [state.longest_wait_yellow] * state.yellow_queue_length

    _, reward, _, _, _ = env.step(action)
    wait_time = env.last_served_wait_times["red"] if action == 0 else env.last_served_wait_times["yellow"]

    # Log to file for monitoring
    log_entry = {
        "timestamp": time.time(),
        "state": state.dict(),
        "computed_free_doctors": computed_free_doctors,
        "action": "RED" if action == 0 else "YELLOW",
        "reward": reward,
        "wait_time": wait_time
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "action": "RED" if action == 0 else "YELLOW",
        "reward": reward,
        "wait_time": wait_time,
        "free_doctors": state.free_doctors
    }
