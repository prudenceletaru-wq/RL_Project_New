import warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained since")
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HospitalEnv(gym.Env):
    def __init__(self, max_steps=30):
        super().__init__()

        self.max_steps = max_steps  # maximum steps per episode

        # Observation space: 10 features
        self.observation_space = spaces.Box(low=0, high=200, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # RED=0, YELLOW=1, GREEN=2

        # Doctors
        self.num_doctors = 3
        self.doctor_timers = np.zeros(self.num_doctors)

        # Queues
        self.red_queue = []
        self.yellow_queue = []
        self.green_queue = []

        # Arrival tracking
        self.arrival_caps = {"red": 3, "yellow": 4, "green": 4}
        self.arrivals_done = {"red": 0, "yellow": 0, "green": 0}

        # Service times (lognormal)
        self.red_mu, self.red_sigma = self._lognormal(15, 5)
        self.yellow_mu, self.yellow_sigma = self._lognormal(10, 2)
        self.green_mu, self.green_sigma = self._lognormal(10, 2)

        # Step counter
        self.current_step = 0

        # Track last 5 actions for GREEN reward
        self.last_5_actions = []

        # Track last served wait times per category
        self.last_served_wait_times = {"red": 0, "yellow": 0, "green": 0}

    # -------------------------------------------
    # Lognormal conversion
    # -------------------------------------------
    def _lognormal(self, mean, std):
        variance = std ** 2
        mu = np.log(mean**2 / np.sqrt(variance + mean**2))
        sigma = np.sqrt(np.log(1 + variance / (mean**2)))
        return mu, sigma

    # -------------------------------------------
    # Reset environment
    # -------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.red_queue = []
        self.yellow_queue = []
        self.green_queue = []
        self.arrivals_done = {"red": 0, "yellow": 0, "green": 0}
        self.doctor_timers[:] = 0
        self.current_step = 0
        self.last_5_actions = []
        self.last_served_wait_times = {"red": 0, "yellow": 0, "green": 0}
        obs = self._get_obs()
        return obs, {}

    # -------------------------------------------
    # Observation
    # -------------------------------------------
    def _get_obs(self):
        return np.array([
            np.sum(self.doctor_timers == 0),  # number of free doctors
            max(self.red_queue) if self.red_queue else 0,  # longest wait in red
            max(self.yellow_queue) if self.yellow_queue else 0,  # longest wait in yellow
            max(self.green_queue) if self.green_queue else 0,  # longest wait in green
            len(self.red_queue),  # red queue length
            len(self.yellow_queue),  # yellow queue length
            len(self.green_queue),  # green queue length
            self.doctor_timers[0],  # doctor 1 busy time left
            self.doctor_timers[1],  # doctor 2 busy time left
            self.doctor_timers[2],  # doctor 3 busy time left
        ], dtype=np.float32)

    # -------------------------------------------
    # Sample service time
    # -------------------------------------------
    def _sample_service(self, action):
        if action == 0:
            return np.random.lognormal(self.red_mu, self.red_sigma)
        elif action == 1:
            return np.random.lognormal(self.yellow_mu, self.yellow_sigma)
        else:
            return np.random.lognormal(self.green_mu, self.green_sigma)

    # -------------------------------------------
    # Add new arrivals
    # -------------------------------------------
    def _add_new_arrivals(self):
        new_red = np.random.poisson(lam=1)
        new_yellow = np.random.poisson(lam=2)
        new_green = np.random.poisson(lam=3)

        self.red_queue.extend([0] * new_red)
        self.yellow_queue.extend([0] * new_yellow)
        self.green_queue.extend([0] * new_green)

        # Increase waiting time of all existing patients
        self.red_queue = [w + 1 for w in self.red_queue]
        self.yellow_queue = [w + 1 for w in self.yellow_queue]
        self.green_queue = [w + 1 for w in self.green_queue]

    # -------------------------------------------
    # Step function
    # -------------------------------------------
    def step(self, action):
        self.current_step += 1

        # Wait until at least one doctor is free
        free_doctors = np.where(self.doctor_timers == 0)[0]
        if len(free_doctors) == 0:
            min_timer = min([t for t in self.doctor_timers if t > 0])
            self.doctor_timers = np.maximum(0, self.doctor_timers - min_timer)
            free_doctors = np.where(self.doctor_timers == 0)[0]

        doctor = free_doctors[0]

        # Map action to queue
        queue_map = {0: ("red", self.red_queue),
                     1: ("yellow", self.yellow_queue),
                     2: ("green", self.green_queue)}
        cat_name, queue = queue_map[action]

        # If queue empty → 0 reward
        if len(queue) == 0:
            reward = 0
        else:
            wait_time = queue.pop(0)
            dt = self._sample_service(action)
            self.doctor_timers[doctor] = dt

            # Advance other doctors
            for i in range(self.num_doctors):
                if i != doctor:
                    self.doctor_timers[i] = max(0, self.doctor_timers[i] - dt)

            # Reward calculation
            reward_map = {"red": 40, "yellow": 20, "green": 5}
            reward = reward_map[cat_name] + 1

            # Dynamic bonus depending on category and threshold
            threshold_times = {"red": 10, "yellow": 30, "green": 60}
            reward_bonus = {"red": 40, "yellow": 30, "green": 10}
            if wait_time <= threshold_times[cat_name]:
                reward += reward_bonus[cat_name]

            # Save last served wait time for this category
            self.last_served_wait_times[cat_name] = wait_time

        # GREEN patient reward for recent service (fairness)
        self.last_5_actions.append(action)
        if len(self.last_5_actions) > 5:
            self.last_5_actions.pop(0)
        if 2 in self.last_5_actions:
            reward += 15

        # Add new arrivals
        self._add_new_arrivals()

        # Truncated if max steps reached
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, False, truncated, {}
