<p align="center">
  <img src="images/baby.gif" alt="An moving image of the albert robot">
</p>

![Version](https://img.shields.io/badge/version-1.0.0-green.svg)

# PDM Group 36 | Undecided Indecision

Welcome to the Planning and Decision Making repo of Group 36. Here you will find the solution for the final project. Happy reading!

## Running the examples

First, create and activate a python virtual environment by running:

```
python -m venv .venv
./.venv/bin/activate
```

> .venv is automatically gitignored in this project. If you want to store the environment in a different folder, you have to add it to the gitignore

Next, install the requirements needed to run this project:

```
pip install -r requirements.txt
```

This will automatically install all modules needed to run the simulation.
These can then be run by:

```
python main.py
```

## Architecture

The project follows a modular architecture with clear separation of concerns:

<p align="center">
  <img src="images/UML.png" alt="UML Class Diagram" width="600">
</p>

### Core Components

- **MissionStateMachine**: Central orchestrator managing state transitions (TUCK, DRIVE, REACH) and coordinating all subsystems
- **PlaygroundEnv**: Environment manager that creates obstacles and provides 2D obstacle data
- **MissionPlanner**: Computes safe base positions and creates mission configurations from environment goals
- **RRTPlanner**: Pure path planning component (Simple/Smooth/Star variants) that computes collision-free paths using RRT algorithms
- **PathFollower**: Control component that generates base velocity commands to follow a given path
- **ArmController**: Simple P-controller using inverse kinematics for arm movements (tucking configuration)
- **MppiArmController**: Advanced MPPI-based controller for collision-aware arm reaching with adaptive exploration
- **PathVisualizer**: Visualization component for displaying paths, obstacles, and goals in 2D
