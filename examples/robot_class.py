import numpy as np
from classes import BaseRobot, FeedbackPolicy

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle


def run_mobile_reacher(n_steps=10000, render=False, goal=True, obstacles=True):    
    model = GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel")       
    # Target obstacle position
    target_obstacle = np.array([2.0, -0.5, 0.15])
    forward_velocity = 0.2  # Base forward velocity (m/s)
    
    # Create FeedbackPolicy
    policy = FeedbackPolicy(
        target_obstacle=target_obstacle,
        forward_velocity=forward_velocity,
        action_size=12,
        step=0
    )

    robot = BaseRobot(policy=policy, model=model)
    
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=[robot.get_model()], render=render, num_sub_steps=200,
    )

    # Add obstacles to the environment
    if obstacles:
        # Add a sphere obstacle directly in the robot's path (will cause collision)
        # Robot starts at [0, 0, 0] and moves forward (positive x) at 0.1 m/s
        sphere_obst_dict = {
            "type": "sphere",
            "movable": False,
            "geometry": {"position": [-0.5, 0.0, 0.2], "radius": 0.25},  # Behind the robot
            "rgba": [0.3, 0.5, 0.6, 1.0],  # Color: [R, G, B, Alpha]
        }
        sphere_obst = SphereObstacle(name="sphere_1", content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)
        
        # Add a box obstacle
        box_obst_dict = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [2.0, -0.5, 0.15],
                "width": 0.3,
                "height": 0.3,
                "length": 0.3,
            },
            "rgba": [0.8, 0.2, 0.2, 1.0],  # Red color
        }
        box_obst = BoxObstacle(name="box_1", content_dict=box_obst_dict)
        env.add_obstacle(box_obst)
        
        # Add a cylinder obstacle
        cylinder_obst_dict = {
            "type": "cylinder",
            "movable": False,
            "geometry": {
                "position": [1.5, 1.0, 0.3],
                "radius": 0.15,
                "height": 0.6,
            },
            "rgba": [0.2, 0.8, 0.2, 1.0],  # Green color
        }
        cylinder_obst = CylinderObstacle(name="cylinder_1", content_dict=cylinder_obst_dict)
        env.add_obstacle(cylinder_obst)
    
    # Create BaseRobot with the policy
    
    ob = env.reset()
    print(f"Initial observation : {ob}")
    history = []
    
    for i in range(n_steps):
        # Update robot state with current observation
        robot.state = ob
        
        # Get action from robot
        action = robot.move()
        
        ob, reward, terminated, truncated, info = env.step(action)  # Apply all 12 actions for this timestep
        robot.update_state(ob)
        history.append(ob)
        
        # Check for collision
        if "Collision" in info:
            print(f"\nCOLLISION DETECTED at step {i}!")
            print(f"   {info['Collision']}")
            break
    env.close()
    return history


if __name__ == "__main__":
    run_mobile_reacher(render=True)