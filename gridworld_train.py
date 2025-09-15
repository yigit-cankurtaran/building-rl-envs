from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gridworld_tut import GridWorldEnv
import os


def train():
    train_env = make_vec_env(GridWorldEnv, n_envs=4)
    eval_env = DummyVecEnv(GridWorldEnv)
    # train and eval envs declared

    os.makedirs("model", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    model_path = "./model"
    log_path = "./log"

    # TODO: might need to ankify EvalCallback internals from previous training files
    eval_callback = EvalCallback(eval_env, log_path, best_model_save_path=model_path)

    model = PPO(
        env=train_env
        # going with the defaults for the rest
        # TODO: tune these according to the outputs we get
    )


if __name__ == "__main__":
    train()
