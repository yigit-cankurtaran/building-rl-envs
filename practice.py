import gymnasium as gym
import numpy as np

"""file used to create anki cards for env code
looking at functions and methods used to create envs with gymnasium.Env"""


# placeholder for a visual RL task
class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        print("env initialized")

        from gymnasium import spaces
        # define discrete action and continuous obs spaces

        self.action_space = spaces.Discrete(4)  # 4 possible actions
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

        self.state = None
        self.steps = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # calls reset from gym.Env, not itself

        self.state = self.observation_space.sample()
        self.steps = 0
        info = {}
        return self.state, info

    def step(self, action):
        # execute action, return (obs,reward,term,trunc,info)

        self.steps += 1

        # TODO: fill these out
        next_state = self.observation_space.sample()  # placeholder
        reward = np.random.random()  # placeholder
        terminated = False  # episode ends when task complete
        truncated = self.steps >= self.max_steps  # episode ends when time limit
        info = {}

        self.state = next_state
        return next_state, reward, terminated, truncated, info

    def render(self, mode="human"):
        # TODO, implement visualization
        pass


env = CustomEnv()
