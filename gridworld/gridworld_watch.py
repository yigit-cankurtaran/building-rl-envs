from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gridworld_tut import GridWorldEnv
from stable_baselines3.common.monitor import Monitor


def watch_trained_agent(ep_count=5):
    model = PPO.load("./model/best_model.zip")
    env = GridWorldEnv(render_mode="human")
    env = Monitor(env)
    rewards, lengths = evaluate_policy(
        model, env, n_eval_episodes=ep_count, render=True, return_episode_rewards=True
    )
    for i in range(
        len(rewards)
    ):  # changed to len(rewards) to only access things that already exist
        print(f"epoch:{i + 1}, reward:{rewards[i]}, length:{lengths[i]}")


if __name__ == "__main__":
    watch_trained_agent()
