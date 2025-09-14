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

        # Dict space gives us structured, human readable observations
        self.observation_space = gym.spaces.Dict(
            {
                # identical constraints bc both grid coords with same valid ranges
                # fully observable, agent knows where it is and where target is
                "agent": gym.spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, (2,), int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)  # 4 actions

        # map action nums to movements on grid
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        def _get_obs(self):
            """translate internal state to observation format
            returns a dict with \"agent\" and \"target\" keys"""
            return {"agent": self._agent_location, "target": self._target_location}

        def _get_info(self):
            """extra info for debugging
            returns a dict with key \"distance\" with l1 distance between agent and input
            """
            return {
                "distance": np.linalg.norm(
                    self._agent_location - self._target_location, ord=1
                )
            }
