import copy
from typing import Tuple
import gym
import numpy as np
import highway_env
from genome import roulette_wheel_selection, crossover, mutation, genome
from population import population
from base_nn import neural_network
from create_network import create_network


class train_genome(genome):

    def get_model(self, input_size, hidden_size, output_size) -> neural_network:
        return create_network(input_size, hidden_size, output_size)

    def run_single(self, environment, n_episodes=10000, render=False) -> Tuple[float, np.array]:
        observation = environment.reset()
        observation = observation["observation"]
        fitness = 0
        for each in range(n_episodes):
            if render:
                environment.render()
            action = self.nn.forward(observation)
            observation, reward, done, info = environment.step(action)
            observation = observation["observation"]
            fitness = reward
            if done:
                break
        return fitness, self.nn.retrieve_weight_and_bias()


def generation(env, old_population, new_population, p_mutation, p_crossover,p_inversion=None):
    for i in range(0, len(old_population) - 1, 2):
        # selection
        parent1 = roulette_wheel_selection(old_population)
        parent2 = roulette_wheel_selection(old_population)

        # crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 p_crossover)

        # mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)


        child1.update_model()
        child2.update_model()

        child1.get_fitness_score(env)
        child2.get_fitness_score(env)

        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2


if __name__ == '__main__':
    env = gym.make('parking-v0')
    env.seed(123)
    population_size = 1000
    no_generation = 1000

    mutation_rate = 0.8
    crossover_rate = 0.5

    input = 6
    hidden_layer = 3
    output = 2

    start_run = population(train_genome(input, hidden_layer, output),
                   population_size, no_generation, mutation_rate, crossover_rate, 0)
    start_run.run(env, generation, verbose=True,
          output_folder='./models/')

    env.close()
