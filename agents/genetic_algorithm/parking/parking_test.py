import gym.wrappers

from mlp import MLP
import pickle
import highway_env

def test_cartpole(nn, file):
    global observation
    nn.load(file)
    for _ in range(5000):
        env.render()
        action = nn.forward(observation)
        observation, reward, done, info = env.step(action)
        observation = observation["observation"]
        if done:
            break


def save_model(nn, filename):
    with open(filename, 'wb') as output:
        pickle.dump(nn, output)


if __name__ == '__main__':
    env = gym.make('parking-v0')
    env.seed(123)
    # env = gym.wrappers.Monitor(env, 'cartpole', video_callable=lambda episode_id: True, force=True)
    observation = env.reset()
    observation = observation["observation"]

    # nn = MLP(4, 2, 1)
    nn = MLP(6, 3, 2)
    # test_cartpole(nn, '../../../models/cartpole/cartpole12-27-2019_20-29_NN=MLPIndividual_POPSIZE=100_GEN'
    #                   '=20_PMUTATION_0.4_PCROSSOVER_0.9.npy')
    # below is when reward is a summation
    test_cartpole(nn,"./models/parking-04-06-2022_00-07_NN=MLPIndividual_POPSIZE=10_GEN=10_PMUTATION_0.8_PCROSSOVER_0.5_I=6_SCORE=-0.11999813473973071.npy")


    env.close()
