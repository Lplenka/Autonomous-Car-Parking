from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from base_nn import neural_network


class genome(ABC):
    def __init__(self, input_size, hidden_size, output_size):
        self.nn = self.get_model(input_size, hidden_size, output_size)
        self.fitness = 0.0
        self.weights_biases: np.array = None

    def get_fitness_score(self, env) -> None:
        self.fitness, self.weights_biases = self.run_single(env)

    def update_model(self) -> None:
        self.nn.weight_bias_reform(self.weights_biases)


    @abstractmethod
    def get_model(self, input_size, hidden_size, output_size) -> neural_network:
        pass

    @abstractmethod
    def run_single(self, env, n_episodes=100000, render=False) -> Tuple[float, np.array]:
        pass


def crossover(parent1_weights_biases: np.array, parent2_weights_biases: np.array, p: float):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)

    if np.random.rand() < p:
        child1_weights_biases[position:], child2_weights_biases[position:] = \
            child2_weights_biases[position:], child1_weights_biases[position:]
    return child1_weights_biases, child2_weights_biases




def mutation(parent_weights_biases: np.array, p: float, scale=10):
    child_weight_biases = np.copy(parent_weights_biases)
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        n = np.random.normal(np.mean(child_weight_biases), np.std(child_weight_biases))
        child_weight_biases[position] = n + np.random.randint(-scale, scale)
    return child_weight_biases


# def ranking_selection(population: List[genome]) -> Tuple[genome, genome]:
#     sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
#     parent1, parent2 = sorted_population[:2]
#     return parent1, parent2


def roulette_wheel_selection(population: List[genome]):
    total_fitness = np.sum([individual.fitness for individual in population])
    selection_probabilities = [individual.fitness / total_fitness for individual in population]
    pick = np.random.choice(len(population), p=selection_probabilities)
    return population[pick]


def statistics(population: List[genome]):
    population_fitness = [individual.fitness for individual in population]
    return np.mean(population_fitness), np.min(population_fitness), np.max(population_fitness)
