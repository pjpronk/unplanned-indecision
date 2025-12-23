import numpy as np
from classes import RRTPlanner, PathFollower, ObstacleManager, PathVisualizer
# Import your new simplified controller
from arm_controller import ArmController 

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv

def run_mobile_reacher(n_steps=10000, render=False, goal=True, obstacles=True):    
    model = GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel")       
    
    # Target position (2D: x, y)
    target_position = np.array([2.0, -1.5])
    forward_velocity = 1.5  # Base forward velocity (m/s)
    action_size = 12
    
    # Create obstacle manager
    obstacle_manager = ObstacleManager()
    obstacles_2d = obstacle_manager.get_2d_obstacles()
    
    # Create RRT Planner
    planner = RRTPlanner(obstacles=obstacles_2d)
    
    # Create PathFollower
    path_follower = PathFollower(
        forward_velocity=forward_velocity,
        waypoint_threshold=0.1
    )
    
    # Create Arm Controller (This uses the "straight up" logic you requested)
    arm_controller = ArmController()
    
    # Create PathVisualizer
    visualizer = PathVisualizer(obstacles_2d, target_position)
    
    # Create URDF environment
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=[model], render=render, num_sub_steps=200,
    )

    if obstacles:
        obstacle_manager.add_to_urdf_env(env)
    
    ob = env.reset()
    history = []
    
    # Plan initial path
    initial_pos = np.array([0.0, 0.0])
    if isinstance(ob, dict) and 'robot_0' in ob:
        initial_pos = np.array(ob['robot_0']['joint_state']['position'][:2])
    
    current_path = planner.plan_path(initial_pos, target_position)
    print(f"Initial path planned with {len(current_path)} waypoints")
    visualizer.show(current_path, current_pos=initial_pos, title="Initial Path Plan")
    
    # --- SIMULATION LOOP ---
    for i in range(n_steps):
        
        # 1. Extract State
        current_pos = np.array([0.0, 0.0])
        current_arm_joints = np.zeros(7)
        
        if isinstance(ob, dict) and 'robot_0' in ob:
            robot_state = ob['robot_0']['joint_state']['position']
            current_pos = np.array(robot_state[:2])
            # Extract arm joints (indices 3 to 9)
            current_arm_joints = np.array(robot_state[3:10])
        
        # 2. Initialize Action Vector (All zeros)
        action = np.zeros(action_size)
        
        # 3. Check Distance to Goal
        distance_to_goal = np.linalg.norm(current_pos - target_position)
        
        if distance_to_goal > 0.15:
            # === STATE 1: DRIVE ===
            # Use your original path follower
            base_action = path_follower.follow(current_pos, current_path, action_size)
            
            # Map the follower's output (size 12) to the action vector
            # Your path_follower returns a full size 12 vector where only 0,1,2 might be set
            action = base_action 
            
            # Explicitly force arm velocity to 0 while driving (safety)
            action[3:10] = 0.0
            
        else:
            # === STATE 2: ARM CONTROL (CANDLE) ===
            # Stop the base
            action[0:3] = 0.0
            
            # Get velocity commands to put arm straight up
            arm_vels = arm_controller.get_target_velocities(current_arm_joints)
            
            # Assign to arm indices
            action[3:10] = arm_vels
            
            if i % 100 == 0:
                print(f"Step {i}: Reaching... Arm Error: {np.linalg.norm(arm_vels):.4f}")

        # 4. Step Environment
        ob, reward, terminated, truncated, info = env.step(action)
        history.append(ob)
        
        if "Collision" in info:
            print(f"\nCOLLISION DETECTED at step {i}!")
            print(f"   {info['Collision']}")
            break
    
    env.close()
    return history

if __name__ == "__main__":
    run_mobile_reacher(render=True)