# grid world tutorial from gymnasium docs

from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    def __init__(self, size: int = 9, render_mode=None):
        """initializing env
        args: size = length of edges of the square grid"""
        self.size = size  # 9x9 by default
        self.render_mode = render_mode

        # init positions, (-1,-1) as uninitialized
        # _ before var name = these are private(internal)
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # for truncation
        self.max_steps = 100
        self.current_step = 0

        # Dict space gives us structured, human readable observations
        self.observation_space = gym.spaces.Dict(
            {
                # identical constraints bc both grid coords with same valid ranges
                # fully observable, agent knows where it is and where target is
                "agent": gym.spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, (2,), int),
            }
        )

        self.action_space = gym.spaces.Discrete(8)  # 8 actions (cardinal + diagonal)

        # map action nums to movements on grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # move right
            1: np.array([0, 1]),  # move up
            2: np.array([-1, 0]),  # move left
            3: np.array([0, -1]),  # move down
            4: np.array([1, 1]),  # northeast
            5: np.array([-1, 1]),  # northwest
            6: np.array([1, -1]),  # southeast
            7: np.array([-1, -1]),  # southwest
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """initialize or restart env to starting state
        args:
        seed=if we want a set seed we can implement this, else None
        options=required for gym envs"""

        # need to first call reset from gym.Env
        # needed for self.np_random, if we don't do this env crashes
        super().reset(seed=seed)

        self.current_step = 0  # reset steps

        # place agent randomly
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location

        # move target away from agent
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        # action = what the agent does, 0-3 because action_to_direction

        self.current_step += 1
        direction = self._action_to_direction[action]

        # update agent position, ensuring it stays in grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # win when agent gets to the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        truncated = self.current_step >= self.max_steps

        reward = 1 if terminated else -0.01

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self, mode=None):
        """render the gridworld environment

        mode (str): "human" prints to console, "rgb_array" returns numpy array, None does nothing
        """
        if mode is None:
            return
        elif mode == "human":
            grid = np.full((self.size, self.size), ".", dtype=str)

            # target (T) and agent (A)
            # numpy arrays use (row, col) but our coordinates are (x, y)
            # flip coordinates when indexing grid
            target_row, target_col = self._target_location[1], self._target_location[0]
            agent_row, agent_col = self._agent_location[1], self._agent_location[0]

            grid[target_row, target_col] = "T"
            grid[agent_row, agent_col] = "A"

            if np.array_equal(self._agent_location, self._target_location):
                grid[agent_row, agent_col] = "W"  # W for win

            # print grid
            print("\n" + "─" * (self.size * 2 + 1))
            for row in grid:
                print("│" + " ".join(row) + "│")
            print("─" * (self.size * 2 + 1))

            # additional info
            print(f"Agent: {self._agent_location}, Target: {self._target_location}")
            print(
                f"Distance: {self._get_info()['distance']:.1f}, Step: {self.current_step}/{self.max_steps}"
            )

        elif mode == "rgb_array":
            # RGB array representation for programmatic use
            rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)

            # white background
            rgb_array.fill(255)

            # target loc in red
            target_row, target_col = self._target_location[1], self._target_location[0]
            rgb_array[target_row, target_col] = [255, 0, 0]  # red

            # agent loc blue
            agent_row, agent_col = self._agent_location[1], self._agent_location[0]
            rgb_array[agent_row, agent_col] = [0, 0, 255]  # blue

            # if agent reaches target, make it green
            if np.array_equal(self._agent_location, self._target_location):
                rgb_array[agent_row, agent_col] = [0, 255, 0]  # green

            return rgb_array

        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """clean up rendering resources if needed
        we don't need it here, but it's required for envs that use GUI frameworks to render
        e.g. we could use this for closing pygame windows, matplotlib figures, release opengl contexts etc."""
        pass
