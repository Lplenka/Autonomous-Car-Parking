# Autonomous Car Parking

This repo contains the code for our AI project titled "AI algorithms for Autonomous Car Parking".


<p align="center">
  <img alt="result video" src="exp_results\results.gif" />
</p>

## Folder Structure
```bash
├── agents
│   ├── reinforcement_learning
│   │   ├── sac_her_agent.py
│   │   ├── a2c_agent.py
│   ├── reinforcement_learning
│   │   ├── imitation_sac_agent.py
│   │   ├── load_imitation.py.py
│   ├── genetic_algorithm
│   │   ├── parking
├── custom_libraries
│   ├── highway-env-custom
│   ├── imitation
├── exp_results
├── notebooks
│   ├── reinforcement_learning
├── requirements.txt
├──.gitignore  
└── Readme.md
```

## Installation

### Custom libraries
We have used a custom ```highway-env``` library and imitation learning latest version in their github repository. To install it, ```cd``` into the folder and run ```pip install -e .```.

### Conda
Move to the root folder
```
conda create --name aiproject python=3.8
conda activate aiproject
pip install -r requirements.txt
```
### Execution
```
conda activate aiproject
python main.py

```

### Experiment

[Highway-env](https://github.com/eleurent/highway-env) environment is used for the experiment. Here parking-env is a continuous action space in which agent decides and dictates the throttle and the steering angle of the vehicle based on various aspects and its observation like its current position along the 2-D coordinates(x and y), its current velocity (v x and v y) and the angle of its wheels. And after each step absolute distance is calculated between the vehicle’s current position and the goal state. The distance is then procured on the weights which then tells how much each element has to be weighed in order to reach or reduce the distance between the two points. This signifies that the agents gains its reward from its absolute distance from the goal while moving in the current direction. Velocity of the vehicle makes sure that the agent is not gaining reward just by standing in a fixed position. Ultimately reward is decided based on velocity and its proximity from the goal state.

Thus vehicle starts with a high negative reward as its away from the goal and static, and then gradually it moves towards the goal and gaining reward by maintaining its velocity and reducing distance obtaining rewards and eventually inching close to 0. The goal state has a reward of 0.12, algorithms tries to get to the long term reward for the agent.

## Algorithms

* Reinforcement Learning: Soft Actor Critic (SAC)
* Imitation Learning: Behaviour Cloning
* Neuroevolution: Genetic Algorithm


## Evaluation Metric

SAC-HER and imitation algorithms were tested for three episodes and genetic
algorithm for one episode. Initial and goal state for SAC-HER and imitation
algorithms were different for each episode where as these two were kept same
for genetic algorithm. The performance of each algorithm is evaluated based on
below evaluation matrices:
– Cumulative reward across all the episodes
– Average velocity across all the episodes

## Results

<p align="center">
  <img alt="result video" src="exp_results\Rewards vs steps.png"  width="300" height="200"/>
  <img alt="result video" src="exp_results\Velocity vs steps.png" width="300" height="200"/ style="padding-left:20px;">
</p>

1) We can observe that the rewards for all three algorithms are gradually increasing with steps taken. Both SAC and imitation algorithm achieves lowest reward of -0.1 as the run progresses whereas genetic algorithm achieves a lowest reward of -0.3 at the end of the run. SAC and imitation algorithm clearly outperforms genetic algorithm and are able  to perform the task efficiently.

2) We can observe that the velocity of the vehicle decreases at the end of run for SAC and imitation algorithm indicating the vehicle is coming to halt once it reaches the parking spot. The fluctuations in the velocity over steps also shows that vehicle is able to adjust its throttle during the travel path. Genetic algorithm shows a contradictory behavior where velocity is not decreasing at the end of the run.

## Key Takeaways
1) Soft actor critic (SAC) with hindsight experience replay significantly outperformed the imitation learning and neuroevolution algorithm in terms of cumulative rewards and average velocity. SAC agent has much smoother movement and generally takes the shortest path.

2) Imitation learning agent has a jerky movement. However, few times it gets stuck when its close to the goal.

3) The Neuroevolution agent takes longer paths with high velocity and higher acceleration. It didn't perform well in the task. The performance of neuroevolution algorithm can be further improved by optimizing the hyperparamters of genetic algorithm and neural network.


#### References 
Genetic algorithm: https://github.com/robertjankowski/ga-openai-gym
