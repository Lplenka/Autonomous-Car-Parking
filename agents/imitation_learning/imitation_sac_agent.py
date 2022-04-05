import gym
import highway_env
import highway_env_custom
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import HerReplayBuffer, SAC

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from sb3_contrib import TQC
import time

# env = gym.make("CartPole-v1")
env_original = gym.make("parking-v0")
env_custom = gym.make("parkingcustom-v0")

def train_expert():
    print("Training a expert.")
    expert = TQC.load("her_car_env", env=env_original)    
    return expert


def sample_expert_transitions():
    expert = train_expert()
    print("Sampling expert transitions.")        
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env_original)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    )
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()
bc_trainer = bc.BC(
    observation_space=env_custom.observation_space,
    action_space=env_custom.action_space,
    demonstrations=transitions,
)

# reward, _ = evaluate_policy(bc_trainer.policy, env_custom, n_eval_episodes=3, render=True)
# print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=100)

reward, _ = evaluate_policy(bc_trainer.policy, env_custom, n_eval_episodes=3, render=True)
print(f"Reward after training: {reward}")

bc_trainer.save_policy("imitation_her_car_env")

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
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()


record_video("parkingcustom-v0", bc_trainer)