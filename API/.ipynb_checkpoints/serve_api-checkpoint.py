from stable_baselines3 import DQN
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

# ---- Loading the Model  ----
model_path = Path(__file__).resolve().parent.parent / "models" / "dqn_hospital_sb3.zip"
model = DQN.load(str(model_path))

# Mapping RL action numbers to readable names
action_map = {
    0: "serve_red",
    1: "serve_yellow",
    }

# ---- Input Schema ----
class Observation(BaseModel):
    state: list  # must contain 8 numbers

# ---- Prediction Endpoint ----
@app.post("/predict")
def predict_action(data: Observation):
    obs = np.array(data.state, dtype=float)

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
