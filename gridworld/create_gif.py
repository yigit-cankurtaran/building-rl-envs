from stable_baselines3 import PPO
from gridworld_tut import GridWorldEnv
import numpy as np
from PIL import Image
import os

def create_gif(num_episodes=3, gif_name="gridworld_training.gif"):
    """Create a GIF from trained agent episodes"""

    # Load trained model
    model = PPO.load("./model/best_model.zip")

    # Create environment with rgb_array mode
    env = GridWorldEnv(render_mode="rgb_array")

    frames = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_frames = []

        while not done:
            # Get current frame
            frame = env.render()
            # Scale up the frame for better visibility (9x9 is tiny)
            frame_scaled = np.repeat(np.repeat(frame, 50, axis=0), 50, axis=1)
            episode_frames.append(frame_scaled)

            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        # Add final frame and hold it longer
        final_frame = env.render()
        final_frame_scaled = np.repeat(np.repeat(final_frame, 50, axis=0), 50, axis=1)
        episode_frames.extend([final_frame_scaled] * 10)  # Hold final frame

        frames.extend(episode_frames)

        # Add separator frames between episodes (black frame)
        if episode < num_episodes - 1:
            black_frame = np.zeros_like(final_frame_scaled)
            frames.extend([black_frame] * 5)

    env.close()

    # Convert to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as GIF
    pil_frames[0].save(
        gif_name,
        save_all=True,
        append_images=pil_frames[1:],
        duration=200,  # 200ms per frame
        loop=0
    )

    print(f"GIF saved as {gif_name}")
    print(f"Total frames: {len(frames)}")

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists("./model/best_model.zip"):
        print("No trained model found! Please run gridworld_train.py first.")
    else:
        create_gif()