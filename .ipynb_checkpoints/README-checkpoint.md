# RL Hospital Patient Triage

## Project Description
This project implements a Reinforcement Learning (RL) agent to optimize patient scheduling in a hospital environment.  
The agent learns to assign doctors to patients of different priority levels (Red and Yellow) to maximize overall efficiency and minimize waiting times.  
The project includes **training**, **inference**, and **deployment** via a FastAPI endpoint.

## Project Structure
```
RL_Project_New/   <-- root directory
│
├── env/           # Custom environment code
│   └── hospital_env.py
│
├── models/        # Saved RL model
│   └── dqn_hospital_sb3.zip
│
├── training/      # RL training and evaluation scripts
│   └── train_dqn.py
│
├── API/           # API inference code
│   └── serve_api.py
│
├── requirements.txt  # Python dependencies with versions
├── Dockerfile        # Dockerfile for building container
└── README.md         # Project description

```

## Installation

**Clone the repository**:

git clone https://github.com/prudenceletaru-wq/RL_Project_New
cd RL_Project_New

* Create and activate a Conda environment:

conda create -n rl_env python=3.12

conda activate rl_env

* Install required packages:
  
pip install -r requirements.txt

* Training the RL Agent

Run the training script to train the DQN agent:

python training/train_dqn.py

The trained model will be saved in the models/ folder.

* Running the API Locally

Start the FastAPI server to serve the trained model:

python API/serve_api.py

Default URL: http://127.0.0.1:8000

API endpoint: /predict

Input: Dictionary of current hospital state

* Example Request (Python):

import requests

state_dict = {
    "free_doctors": 2,
    "longest_wait_red": 8,
    "longest_wait_yellow": 12,
    "red_queue_length": 4,
    "yellow_queue_length": 5,
    "doctor1_busy_time": 0,
    "doctor2_busy_time": 4,
    "doctor3_busy_time": 2
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"state": state_dict}
)


print(response.json())

Example Response:

{
  "action": 0,
  "meaning": "serve_red"
}

Action values:

0 → Serve Red patient

1 → Serve Yellow patient

* Web-based Testing 

Deployed on Render. Open the interactive API docs:

https://rl-hospital-api.onrender.com/docs

Click on /predict

Click Try it out

Enter a JSON dictionary for state (like the example below):

{
  "state": {
    "free_doctors": 2,
    "longest_wait_red": 8,
    "longest_wait_yellow": 12,
    "red_queue_length": 4,
    "yellow_queue_length": 5,
    "doctor1_busy_time": 0,
    "doctor2_busy_time": 4,
    "doctor3_busy_time": 2
  }
}

Click Execute to see the predicted action.

* Deployment

This project uses Docker for deployment.

Dockerfile is included in the root directory.

To deploy on Render:

Push the repo to GitHub

Connect the repo to Render

Configure as a Docker Web Service

Render automatically builds and deploys the container

* Dependencies

Key libraries (also listed in requirements.txt):

numpy

gymnasium

stable-baselines3

torch

matplotlib

fastapi

pydantic

uvicorn
