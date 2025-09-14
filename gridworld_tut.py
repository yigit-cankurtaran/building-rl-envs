# grid world tutorial from gymnasium docs

from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    def __init__(self, size: int = 5):
        self.size = size  # 5x5 by default

        # init positions, (-1,-1) as uninitialized
        # _ before var name = these are private(internal)
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
