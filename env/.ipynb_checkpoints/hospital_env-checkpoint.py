import warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained since")
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HospitalEnv(gym.Env):
    def __init__(self, max_steps=30):
        super().__init__()

        self.max_steps = max_steps  # maximum steps per episode

        # Observation space: 8 features (free doctors + red/yellow queues + doctor timers)
        self.observation_space = spaces.Box(low=0, high=200, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0=RED, 1=YELLOW

        # Number of doctors and service times
        self.num_doctors = 3
        self.doctor_timers = np.zeros(self.num_doctors)

        # Queues
        self.red_queue = []
        self.yellow_queue = []

        # Arrival tracking (Poisson lambda)
        self.arrival_lam = {"red": 0.5, "yellow": 1}

        # Service times (lognormal)
        self.red_mu, self.red_sigma = self._lognormal(15, 5)
        self.yellow_mu, self.yellow_sigma = self._lognormal(10, 2)

        # Step counter
        self.current_step = 0

        # Last served wait times per category
        self.last_served_wait_times = {"red": 0, "yellow": 0}

    # -------------------------------------------
    # Lognormal conversion
    # -------------------------------------------
    def _lognormal(self, mean, std):
        variance = std ** 2
        mu = np.log(mean**2 / np.sqrt(variance + mean**2))
        sigma = np.sqrt(np.log(1 + variance / (mean**2)))
        return mu, sigma

    # -------------------------------------------
    # Resetting environment
    # -------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.red_queue = []
        self.yellow_queue = []
        self.doctor_timers[:] = 0
        self.current_step = 0
        self.last_served_wait_times = {"red": 0, "yellow": 0}
        obs = self._get_obs()
        return obs, {}

    # -------------------------------------------
    # Observation
    # -------------------------------------------
    def _get_obs(self):
        return np.array([
            np.sum(self.doctor_timers == 0),         # number of free doctors
            max(self.red_queue) if self.red_queue else 0,   # longest wait in red
            max(self.yellow_queue) if self.yellow_queue else 0,  # longest wait in yellow
            len(self.red_queue),                      # red queue length
            len(self.yellow_queue),                   # yellow queue length
            self.doctor_timers[0],                    # doctor 1 busy time left
            self.doctor_timers[1],                    # doctor 2 busy time left
            self.doctor_timers[2],                    # doctor 3 busy time left
        ], dtype=np.float32)

    # -------------------------------------------
    # Sample service time
    # -------------------------------------------
    def _sample_service(self, action):
        if action == 0:
            return np.random.lognormal(self.red_mu, self.red_sigma)
        else:
            return np.random.lognormal(self.yellow_mu, self.yellow_sigma)

    # -------------------------------------------
    # Adding new arrivals
    # -------------------------------------------
    def _add_new_arrivals(self):
        new_red = np.random.poisson(lam=self.arrival_lam["red"])
        new_yellow = np.random.poisson(lam=self.arrival_lam["yellow"])

        self.red_queue.extend([0] * new_red)
        self.yellow_queue.extend([0] * new_yellow)

        # Incrementing waiting time for all patients
        self.red_queue = [w + 1 for w in self.red_queue]
        self.yellow_queue = [w + 1 for w in self.yellow_queue]

    # -------------------------------------------
    # Step function
    # -------------------------------------------
    def step(self, action):
        self.current_step += 1

        # Ensure at least one doctor is free
        free_doctors = np.where(self.doctor_timers == 0)[0]
        if len(free_doctors) == 0:
            min_timer = min([t for t in self.doctor_timers if t > 0])
            self.doctor_timers = np.maximum(0, self.doctor_timers - min_timer)
            free_doctors = np.where(self.doctor_timers == 0)[0]

        doctor = free_doctors[0]

        # Mapping action to queue
        queue_map = {0: ("red", self.red_queue), 1: ("yellow", self.yellow_queue)}
        cat_name, queue = queue_map[action]

        # ------------------------------
        # Penalty if queue is empty
        # ------------------------------
        if len(queue) == 0:
            penalty = -20
            self._add_new_arrivals()
            truncated = self.current_step >= self.max_steps
            return self._get_obs(), penalty, False, truncated, {"message": f"Selected empty {cat_name} queue"}

        # ------------------------------
        # Normal service
        # ------------------------------
        wait_time = queue.pop(0)
        dt = self._sample_service(action)
        self.doctor_timers[doctor] = dt

        # Advancing other doctors
        for i in range(self.num_doctors):
            if i != doctor:
                self.doctor_timers[i] = max(0, self.doctor_timers[i] - dt)

        # Reward calculation
        reward_map = {"red": 40, "yellow": 30}
        reward = reward_map[cat_name] + 1

        # Bonus for serving quickly
        threshold_times = {"red": 15, "yellow": 30}
        reward_bonus = {"red": 35, "yellow": 30}
        if wait_time <= threshold_times[cat_name]:
            reward += reward_bonus[cat_name]

        # Saving last served wait time
        self.last_served_wait_times[cat_name] = wait_time

        # Adding new arrivals
        self._add_new_arrivals()

        # Episode truncation
        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, False, truncated, {}
