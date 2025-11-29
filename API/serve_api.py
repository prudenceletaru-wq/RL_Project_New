from stable_baselines3 import DQN
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import Dict

app = FastAPI()

# ---- Loading the Model ----
model_path = Path(__file__).resolve().parent.parent / "models" / "dqn_hospital_sb3.zip"
model = DQN.load(str(model_path))

# Mapping RL action numbers to readable names
action_map = {
    0: "serve_red",
    1: "serve_yellow",
}

# ---- Input Schema ----
class Observation(BaseModel):
    state: Dict[str, int]  # dictionary with feature names

# ---- Prediction Endpoint ----
@app.post("/predict")
def predict_action(data: Observation):
    # Define the correct order your RL model expects
    ordered_keys = [
        "free_doctors",
        "longest_wait_red",
        "longest_wait_yellow",
        "red_queue_length",
        "yellow_queue_length",
        "doctor1_busy_time",
        "doctor2_busy_time",
        "doctor3_busy_time"
    ]

    # Convert dictionary to list in correct order
    try:
        obs = np.array([data.state[key] for key in ordered_keys], dtype=float)
    except KeyError as e:
        return {
            "error": f"Missing required key in state: {e}"
        }

    # Checking if all queues are empty
    if obs[3] == 0 and obs[4] == 0:
        return {
            "action": None,
            "meaning": "No patients in queues",
            "message": "API detected empty queues, no action taken"
        }

    # Otherwise, predicting action with RL model
    action, _ = model.predict(obs, deterministic=True)
    readable = action_map[int(action)]
    return {"action": int(action), "meaning": readable}
