import numpy as np
from classes import RRTPlanner, PathFollower, ObstacleManager, PathVisualizer

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def run_mobile_reacher(n_steps=10000, render=False, goal=True, obstacles=True):    
    model = GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel")       
    
    # Target position (2D: x, y)
    target_position = np.array([2.0, -1.5])
    forward_velocity = 0.5  # Base forward velocity (m/s)
    action_size = 12
    
    # Create obstacle manager
    obstacle_manager = ObstacleManager()
    
    # Get 2D obstacles for path planning
    obstacles_2d = obstacle_manager.get_2d_obstacles()
    
    # Create RRT Planner
    planner = RRTPlanner(
        obstacles=obstacles_2d
    )
    
    # Create PathFollower
    path_follower = PathFollower(
        forward_velocity=forward_velocity,
        waypoint_threshold=0.1
    )
    
    # Create PathVisualizer
    visualizer = PathVisualizer(obstacles_2d, target_position)
    
    # Create URDF environment
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=[model], render=render, num_sub_steps=200,
    )

    # Add obstacles to the URDF environment
    if obstacles:
        obstacle_manager.add_to_urdf_env(env)
    
    ob = env.reset()
    history = []
    
    # Plan initial path and visualize
    initial_pos = np.array([0.0, 0.0])  # Default start position
    if isinstance(ob, dict) and 'robot_0' in ob:
        initial_pos = np.array(ob['robot_0']['joint_state']['position'][:2])
    
    # Plan path once from start to goal
    current_path = planner.plan_path(initial_pos, target_position)
    print(f"Initial path planned with {len(current_path)} waypoints")
    
    # Visualize the path before starting simulation
    visualizer.show(current_path, current_pos=initial_pos, title="Initial Path Plan")
    
    for i in range(n_steps):
        
        # Extract current 2D position from observation
        current_pos = np.array([0.0, 0.0])
        if isinstance(ob, dict):
            if 'robot_0' in ob:
                robot_state = ob['robot_0']
                current_pos = np.array(robot_state['joint_state']['position'][:2])
        
        # Get action from path follower
        action = path_follower.follow(current_pos, current_path, action_size)
        
        # Step the environment
        ob, reward, terminated, truncated, info = env.step(action)
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