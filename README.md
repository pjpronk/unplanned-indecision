<p align="center">
  <img src="images/albert.gif" alt="An moving image of the albert robot">
</p>


![Version](https://img.shields.io/badge/version-1.0.0-green.svg)
# PDM Group 36 | Undecided Indecision

Welcome to the Planning and Decision making repo of Group 36. Here you will find the solution for the final project. Happy reading!

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

- **RRTPlanner**: Pure path planning component that computes collision-free paths from start to target using RRT algorithm
- **PathFollower**: Control component that generates robot actions to follow a given path
- **ObstacleManager**: Manages 3D obstacles and provides 2D projections for path planning
- **PathVisualizer**: Visualization component for displaying paths and obstacles in 2D
