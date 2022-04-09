from genetic_algorithm.parking.create_network import create_network
from utils import generate_graphs
import numpy as np
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from gym.wrappers import RecordVideo
from tqdm.notebook import trange
import gym
import highway_env
# import highway_env_custom
# from imitation.algorithms import bc
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import time

sac_episode_dict = {}
imit_episode_dict = {}
gen_episode_dict = {}


def InitialiseModels(env):
    model_class = SAC
    model_sac = model_class.load('reinforcement_learning/her_car_env', env=env)

    # policy = bc.reconstruct_policy("imitation_her_car_env")

    mlp_model = create_network(6, 3, 2)
    mlp_model.load(
        './genetic_algorithm/parking/models/parking_genetic_-0.07105057707839282.npy')
    return model_sac, mlp_model


def RenderSac(env, model):
    env = RecordVideo(env, video_folder='./reinforcement_learning/videos',
                      episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    for episode in trange(3, desc="Test episodes"):
        sac_velocities = []
        sac_rewards = []
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            v_x, v_y = obs["observation"][2], obs["observation"][3]
            sac_velocities.append(np.sqrt(np.square(v_x) + np.square(v_y)))
            sac_rewards.append(reward)
        sac_episode_dict[episode+1] = [sac_velocities, sac_rewards]
    env.close()


def RenderImitation(policy):

    def record_video(env_id, model, video_length=600, prefix=str(int((time.time()))), video_folder='./imitation_learning/videos'):
        eval_env = DummyVecEnv([lambda: gym.make(env_id)])
        eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                    record_video_trigger=lambda step: step == 0, video_length=video_length,
                                    name_prefix=prefix)

        for episode in trange(3, desc="Test episodes"):
            imi_velocities = []
            imi_rewards = []
            obs = eval_env.reset()
            for _ in range(video_length):
                action, _ = model.predict(obs)
                obs, rewards, done, _ = eval_env.step(action)
                v_x, v_y = obs[2], obs[3]
                imi_velocities.append(np.sqrt(np.square(v_x) + np.square(v_y)))
                imi_rewards.append(rewards)
            imit_episode_dict[episode+1] = [imi_velocities, imi_rewards]

        eval_env.close()
    record_video("parkingcustom-v0", policy)


def RenderGenetic(env, mlp_model):
    env = RecordVideo(env, video_folder='./genetic_algorithm/parking/videos',
                      episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    gen_velocities = []
    gen_rewards = []
    env.seed(123)
    observation = env.reset()
    observation = observation["observation"]
    for _ in range(58):
        env.render()
        action = mlp_model.forward(observation)
        observation, reward, done, info = env.step(action)
        observation = observation["observation"]
        v_x, v_y = observation[2], observation[3]
        gen_velocities.append(np.sqrt(np.square(v_x) + np.square(v_y)))
        gen_rewards.append(reward)
        if done:
            break
    gen_episode_dict[1] = [gen_velocities, gen_rewards]
    env.close()


def Generate_Graphs():
    gen_graphs = generate_graphs()
    data_dict = {}
    data_dict['SAC'] = sac_episode_dict
    data_dict['IMI'] = imit_episode_dict
    data_dict['GEN'] = gen_episode_dict
    gen_graphs.process_data(data_dict)


def main():
    env = gym.make("parking-v0")
    model_sac, model_gen = InitialiseModels(env)

    RenderSac(env, model_sac)
    # RenderImitation(env, policy)
    RenderGenetic(env, model_gen)
    Generate_Graphs()


if __name__ == '__main__':
    main()
