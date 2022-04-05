import gym
import highway_env
import highway_env_custom
from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import time

env_original = gym.make("parking-v0")
env_custom = gym.make("parkingcustom-v0")

policy = bc.reconstruct_policy("imitation_her_car_env")

# reward, _ = evaluate_policy(policy, env_custom, n_eval_episodes=3, render=True)
# print(f"Reward after training: {reward}")




def record_video(env_id, model, video_length=600, prefix=str(int((time.time()))), video_folder='./'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make(env_id)])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, done, _ = eval_env.step(action)
    print(done)
    

  # Close the video recorder
  eval_env.close()


record_video("parkingcustom-v0", policy)