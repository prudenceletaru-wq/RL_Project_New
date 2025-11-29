import warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained since")
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from hospital_env import HospitalEnv  # replace with your actual path

# Creating and wrapping environment
env = HospitalEnv()
env = Monitor(env)

# Convergence check function
def check_convergence(mean_rewards, threshold=10):
    """
    Checks if the last two rewards differ by less than threshold.
    Returns True = converged.
    """
    if len(mean_rewards) < 2:
        return False
    return abs(mean_rewards[-1] - mean_rewards[-2]) < threshold

# Loading the trained model
model = DQN.load("../models/dqn_hospital_sb3", env=env)

# Evaluating over timesteps
mean_rewards_history = []
timesteps_history = []

for t in range(0, 200000, 10000):  
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    mean_rewards_history.append(mean_reward)
    timesteps_history.append(t)

    print(f"At {t} timesteps → mean reward: {mean_reward:.2f}")

    if check_convergence(mean_rewards_history):
        print(f"\nModel converged at ~{t} timesteps.\n")
        break
