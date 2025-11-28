# RL Hospital Patient Triage

## Project Description
This project implements a **Reinforcement Learning (RL) agent** to optimize patient triage in a hospital environment.  
The agent learns to assign doctors to patients of different priority levels (Red, Yellow, Green) to **maximize overall efficiency and minimize waiting times**.  
The project includes training, inference, and deployment via a **FastAPI endpoint**.

---

## Project Structure

RL_Hospital_Project/
│
├── env/ # Custom environment code
│ └── hospital_env.py
│
├── models/ # Saved RL models
│ └── dqn_hospital_sb3
│
├── training/ # RL training scripts
│ └── train_dqn.py
│
├── deployment/ # API / Inference code
│ └── serve_api.py
│
├── requirements.txt # Python dependencies with versions
└── README.md # This file


---

## Installation

1. **Clone the repository**:

```bash
git clone <your-repo-url>
cd RL_Hospital_Project


Create and activate a Conda environment:

conda create -n rl_env python=3.12
conda activate rl_env


Install required packages:

pip install -r requirements.txt

Usage
Training the RL Agent

Run the training script to train the DQN agent:

python training/train_dqn.py


The trained model will be saved in the models/ folder.

Running the API

Start the FastAPI server to serve the trained model:

python deployment/serve_api.py


Default URL: http://127.0.0.1:8000

API endpoint: /predict

Input: JSON observation of current hospital state

Example Request:

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"state": [3, 10, 0, 0, 3, 9, 0, 0, 0, 0]}'


Response:

{
  "action": 1
}


action values:

0 → Red patient

1 → Yellow patient

2 → Green patient

Testing Q-values interactively

You can also check what the RL agent predicts for any state:

import numpy as np
import torch
from stable_baselines3 import DQN

model = DQN.load(r"C:\Users\Prudence Letaru\Desktop\RL_Project_New\models\dqn_hospital_sb3")
obs = np.array([3,10,0,0,3,9,0,0,0,0], dtype=np.float32)
obs_tensor = torch.tensor(obs).unsqueeze(0)

q_values = model.q_net(obs_tensor)
print("Q-values for [Red, Yellow, Green]:", q_values.detach().numpy())
print("DQN chooses action:", ["Red","Yellow","Green"][q_values.argmax().item()])

Dependencies

Key libraries (also listed in requirements.txt):

numpy

gymnasium

stable-baselines3

torch

matplotlib

fastapi

pydantic

uvicorn