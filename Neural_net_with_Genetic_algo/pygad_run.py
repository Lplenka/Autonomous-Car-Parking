import gym
import highway_env
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pygad.kerasga
import pygad


def fitness_func(solution, sol_idx):
    global keras_ga, model, observation_space_size, env

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
        model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    # play game
    observation = env.reset()
    sum_reward = 0
    done = False
    c = 0
    while (not done) and c < 1000:
        state = np.reshape(observation["observation"], [1, 6])
        q_values = model.predict(state)
        action = q_values[0]
        observation_next, reward, done, info = env.step(action)
        observation = observation_next
        sum_reward = reward
        c += 1

    return sum_reward


def callback_generation(ga_instance):
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))


# env = gym.make("CartPole-v1")
# #env = gym.make("parking-v0")
# observation_space_size = env.observation_space.shape
# print(observation_space_size)
# action_space_size = env.action_space
# print(action_space_size)

#env = gym.make("CartPole-v1")
env = gym.make("parking-v0")
env.config['duration'] = 1000
observation_space_size = 6
action_space_size = 2

model = Sequential()
model.add(Dense(16, input_shape=(observation_space_size,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(action_space_size, activation='linear'))
model.summary()

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)
# print(keras_ga.population_weights)
# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 5  # Number of generations.
# Number of solutions to be selected as parents in the mating pool.
num_parents_mating = 4
# Initial population of network weights
initial_population = keras_ga.population_weights
parent_selection_type = "rank"  # Type of parent selection.
crossover_type = "uniform"  # Type of the crossover operator.
mutation_type = "inversion"  # Type of the mutation operator.
# Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
mutation_percent_genes = 10
# Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
keep_parents = -1

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(
    title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(
    solution_idx=solution_idx))

model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
    model=model, weights_vector=solution)
model.set_weights(weights=model_weights_matrix)
model.save("parking_weights")

model = tensorflow.keras.models.load_model('parking_weights')

env = gym.make("parking-v0")
observation_space_size = 6
observation = env.reset()
done = False
sum_reward = 0
c = 0
while not done:
    env.render()
    state = np.reshape(observation["observation"], [1, 6])
    q_values = model.predict(state)
    action = q_values[0]
    print('action : ', action)
    observation, reward, done, info = env.step(action)
    # print('done : ', done)
    observation = observation
    sum_reward += reward
    c += 1
    print("Step: {step} sum_reward: {sum_reward}".format(
        step=c, sum_reward=reward))

env.close()
