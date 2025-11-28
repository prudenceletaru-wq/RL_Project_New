from stable_baselines3 import DQN
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

# ---- Load Model Correctly (Docker-safe path) ----
model_path = Path(__file__).resolve().parent.parent / "models" / "dqn_hospital_sb3.zip"
model = DQN.load(str(model_path))

# Map RL action numbers to readable names
action_map = {
    0: "serve_red",
    1: "serve_yellow",
    2: "serve_green"
}

# ---- Input Schema ----
class Observation(BaseModel):
    state: list  # must contain 10 numbers

# ---- Prediction Endpoint ----
@app.post("/predict")
def predict_action(data: Observation):
    obs = np.array(data.state, dtype=float)
    action, _ = model.predict(obs, deterministic=True)
    readable = action_map[int(action)]
    return {"action": int(action), "meaning": readable}
