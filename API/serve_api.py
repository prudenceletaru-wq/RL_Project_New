from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import os
import time
from stable_baselines3 import DQN
from hospital_env import HospitalEnv

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
MODEL_PATH = "models/dqn_hospital_sb3"
model = DQN.load(MODEL_PATH)

# -------------------------------
# Logging location for monitoring
# -------------------------------
LOG_FILE = "api_logs.json"

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
    obs = state_to_obs(request.state)

    # Predict action
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    # Compute reward and wait time using environment logic
    # Here we simulate an environment step
    env = HospitalEnv()
    # Set environment state manually
    env.doctor_timers = np.array([request.state.doctor1_busy_time,
                                  request.state.doctor2_busy_time,
                                  request.state.doctor3_busy_time], dtype=np.float32)
    env.red_queue = [request.state.longest_wait_red]*request.state.red_queue_length
    env.yellow_queue = [request.state.longest_wait_yellow]*request.state.yellow_queue_length

    _, reward, _, _, _ = env.step(action)
    wait_time = env.last_served_wait_times["red"] if action == 0 else env.last_served_wait_times["yellow"]

    # Log to file for monitoring
    log_entry = {
        "timestamp": time.time(),
        "state": request.state.dict(),
        "action": "RED" if action == 0 else "YELLOW",
        "reward": reward,
        "wait_time": wait_time
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "action": "RED" if action == 0 else "YELLOW",
        "reward": reward,
        "wait_time": wait_time
    }
