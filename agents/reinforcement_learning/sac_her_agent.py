import gym
import highway_env
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from gym.wrappers import RecordVideo
from tqdm.notebook import trange
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

# # Initialize the model
# model = model_class(
#     "MultiInputPolicy",
#     env,
#     replay_buffer_class=HerReplayBuffer,
#     # Parameters for HER
#     replay_buffer_kwargs=dict(
#         n_sampled_goal=n_sampled_goal,
#         goal_selection_strategy=goal_selection_strategy,
#         online_sampling=online_sampling,
#         max_episode_length=max_episode_length,
#     ),
#     verbose=1,
# )

# # Train the model
# model.learn(100000)

# model.save("./her_car_env")
# # Because it needs access to `env.compute_reward()`
# # HER must be loaded with the env
# model = model_class.load('./her_car_env', env=env)

# obs = env.reset()

# # Evaluate the agent
# episode_reward = 0
# for _ in range(100):
#   action, _ = model.predict(obs)
#   obs, reward, done, info = env.step(action)
#   env.render()
#   episode_reward += reward
#   if done or info.get('is_success', False):
#     print("Reward:", episode_reward, "Success?", info.get('is_success', False))
#     episode_reward = 0.0
#     obs = env.reset()

model = model_class.load('./her_car_env', env=env)
env = gym.make("parking-v0")
her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future', online_sampling=True, max_episode_length=100)
# # You can replace TQC with SAC agent
# model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
#             replay_buffer_kwargs=her_kwargs, verbose=1, buffer_size=int(1e6),
#             learning_rate=1e-3,
#             gamma=0.95, batch_size=1024, tau=0.05,
#             policy_kwargs=dict(net_arch=[512, 512, 512]))
# model.learn(int(5e4))
# model.learn(int(1000))


env = gym.make("parking-v0")

env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda e: True)
env.unwrapped.set_record_video_wrapper(env)
for episode in trange(3, desc="Test episodes"):
    obs, done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # print("observation\n")
        # print(obs)
        # print("reward\n")
        # print(reward)
        # print("done\n")
        # print(done)
env.close()