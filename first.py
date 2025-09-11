import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # define action and obs spaces
        self.action_space = spaces.Discrete(4)  # 4 possible actions
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
