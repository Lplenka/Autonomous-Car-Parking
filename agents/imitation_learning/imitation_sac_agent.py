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
bc_trainer.train(n_epochs=300)

reward, _ = evaluate_policy(bc_trainer.policy, env_custom, n_eval_episodes=3, render=True)
print(f"Reward after training: {reward}")

bc_trainer.save_policy("imitation_her_car_env")