import copy
from typing import Callable
import numpy as np
import torch
from genome import statistics


class population:
    def __init__(self, individual, pop_size, max_generation, mutation_rate, crossover_rate, inversion_rate):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.inversion_rate = inversion_rate
        self.old_population = [copy.copy(individual) for _ in range(pop_size)]
        self.new_population = []

    def set_population(self, population: list):
        self.old_population = population

    def run(self, env, run_generation: Callable, verbose=False, output_folder=None):
        best_model = sorted(self.old_population, key=lambda ind: ind.fitness, reverse=True)[0]
        for i in range(self.max_generation):
            [p.get_fitness_score(env) for p in self.old_population]

            self.new_population = [None for _ in range(self.pop_size)]
            run_generation(env,
                           self.old_population,
                           self.new_population,
                           self.mutation_rate,
                           self.crossover_rate,
                           self.inversion_rate)

            if verbose:
                self.show_stats(i)

            self.update_old_population()

            new_best_model = self.get_best_model_parameters()

            if new_best_model.fitness > best_model.fitness:
                print('Saving new best model with fitness: '+str(new_best_model.fitness))
                self.save_model_parameters(output_folder, i)
                best_model = new_best_model

        if output_folder:
            self.save_model_parameters(output_folder, self.max_generation)


    def show_stats(self, n_gen):
        mean, min, max = statistics(self.new_population)
        print("No. generation: "+str(n_gen+1) + ', mean: '+str(mean) + ' , min: '+str(min)+' ,max: '+str(max))


    def update_old_population(self):
        self.old_population = copy.deepcopy(self.new_population)

    def save_model_parameters(self, output_folder, iterations):
        best_model = self.get_best_model_parameters()
        file_name = 'parking_genetic_'+str(best_model.fitness)+'.npy'
        output_filename = output_folder + '-' + file_name
        np.save(output_filename, best_model.weights_biases)

    def get_best_model_parameters(self) -> np.array:
        return sorted(self.new_population, key=lambda ind: ind.fitness, reverse=True)[0]


