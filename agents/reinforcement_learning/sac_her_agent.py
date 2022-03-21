import gym
import highway_env
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
# from stable_baselines3.ddpg import NormalActionNoise

env = gym.make("parking-v0")

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

model_class = SAC  # works also with DQN, DDPG and TD3
N_BITS = 15

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = N_BITS

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
)

# Train the model
model.learn(1000)

model.save("./her_car_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = model_class.load('./her_car_env', env=env)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(100):
  action, _ = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
  episode_reward += reward
  if done or info.get('is_success', False):
    print("Reward:", episode_reward, "Success?", info.get('is_success', False))
    episode_reward = 0.0
    obs = env.reset()