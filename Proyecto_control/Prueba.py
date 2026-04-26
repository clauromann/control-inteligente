import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./videos")

obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()