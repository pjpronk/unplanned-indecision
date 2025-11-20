import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle


def run_mobile_reacher(n_steps=10000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel"),
    ]
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=robots, render=render, num_sub_steps=200,
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
    
    
    # Action array: 12 values, one per joint (velocity commands in vel mode)
    # [mobile_x, mobile_y, mobile_theta, panda_joint1-7, finger1, finger2]
    action = np.zeros(env.n())
    action[5] = -0.1     # Arm joint 3 velocity (rad/s) - effectively zero
    action[-1] = 3.5     # Gripper finger 2 velocity (rad/s)
    
    # Target obstacle position
    target_obstacle = np.array([2.0, -0.5, 0.15])
    forward_velocity = 0.2  # Base forward velocity (m/s)
    
    ob = env.reset()
    print(f"Initial observation : {ob}")
    history = []
    
    # Track robot position
    robot_pos = np.array([0.0, 0.0, 0.0])
    
    for i in range(n_steps):
        # Get current robot position from observation
        # Observation is a dict with 'position' and 'velocity' keys
        if isinstance(ob, dict) and 'position' in ob:
            position_array = ob['position']
            if len(position_array) >= 3:
                robot_pos[0] = position_array[0]  # x position
                robot_pos[1] = position_array[1]  # y position
                robot_pos[2] = position_array[2]  # z position
        
        # Calculate direction to target obstacle
        direction = target_obstacle[:2] - robot_pos[:2]  # 2D direction vector (x, y)
        distance = np.linalg.norm(direction)
        
        if distance > 0.05:  # If not very close yet
            # Normalize direction and scale by desired speed
            direction_normalized = direction / distance
            action[0] = direction_normalized[0] * forward_velocity  # x velocity
            action[1] = direction_normalized[1] * forward_velocity  # y velocity
        else:
            # Very close, just keep moving forward
            action[0] = forward_velocity
            action[1] = 0.0
        
        # Alternate gripper actions every 100 steps
        if (int(i / 100)) % 2 == 0:
            action[-1] = -0.01  # Close finger 2
            action[-2] = -0.01  # Close finger 1
        else:
            action[-1] = 0.01   # Open finger 2
            action[-2] = 0.01   # Open finger 1
        ob, reward, terminated, truncated, info = env.step(action)  # Apply all 12 actions for this timestep
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