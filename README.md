# CS7IS2_AI_group_project

This repo contains the code for our AI project titled "AI algorithms for Autonomous Car Parking".

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
├── minutes
├── report
├── requirements.txt
├──.gitignore  
└── Readme.md
```

## Installation

#### Custom libraries
We have used a custom ```highway-env``` library and imitation learning latest version in their github repository. To install it, ```cd``` into the folder and run ```pip install -e .```.

#### Conda
Move to the root folder
```
conda create --name aiproject python=3.8
conda activate aiproject
pip install -r requirements.txt
```
#### Execution
```
conda activate aiproject
python main.py
```

#### References 
Genetic algorithm: https://github.com/robertjankowski/ga-openai-gym