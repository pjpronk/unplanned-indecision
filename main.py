import numpy as np
import pybullet as p

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from classes import (RRTPlanner, PathFollower, PathVisualizer, 
                     PlaygroundEnv, ArmController, MppiArmController, PandaConfig,
                     MissionStateMachine, MissionPlanner, MissionConfig)


# --- Helper Functions ---
def get_arm_joint_states(robot_id: int, robot_config) -> np.ndarray:
    """Extract current arm joint positions from PyBullet."""
    return np.array([
        p.getJointState(robot_id, idx)[0] 
        for idx in robot_config.ARM_JOINT_INDICES
    ])


def get_state_from_observation(obs: dict, robot_config) -> tuple:
    """Extract base (x,y) and arm joints from observation."""
    if isinstance(obs, dict) and robot_config.ROBOT_NAME in obs:
        state = obs[robot_config.ROBOT_NAME]["joint_state"]["position"]
        base_xy = np.array(state[:2])
        arm_q = np.array(state[robot_config.ARM_OBS_SLICE])
        return base_xy, arm_q
    return np.zeros(2), np.zeros(7)


def create_target_marker(position: list, size: float = 0.05, 
                         color: tuple = (1.0, 0.0, 1.0, 0.6)) -> int:
    """Create a target marker in PyBullet with collision shape."""
    half_size = size / 2.0
    
    # Create both visual and collision shapes
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half_size, half_size, half_size],
        rgbaColor=color
    )
    
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half_size, half_size, half_size]
    )
    
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape_id,
        baseCollisionShapeIndex=collision_shape_id,
        basePosition=position,
        baseOrientation=[0, 0, 0, 1]
    )


def remove_collided_obstacle(collision_info: str, robot_id: int) -> bool:
    """
    Remove the obstacle that the robot collided with.
    
    Args:
        collision_info: Collision string from env info
        robot_id: PyBullet body ID of the robot
        
    Returns:
        True if an obstacle was removed
    """
    num_bodies = p.getNumBodies()
    
    for body_id in range(num_bodies):
        if body_id == robot_id:
            continue
        
        # Check if this body is in contact with robot
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=body_id)
        if len(contacts) > 0:
            # Get obstacle name for logging
            try:
                body_name = p.getBodyInfo(body_id)[1].decode("utf-8")
                print(f"  Removing obstacle '{body_name}' (ID: {body_id})")
            except Exception:
                print(f"  Removing obstacle (ID: {body_id})")
            
            # Remove the obstacle
            p.removeBody(body_id)
            return True
    
    return False


def tuck_arm_to_candle(env, robot_id: int, arm_controller: ArmController, 
                       robot_config, threshold: float = 0.1, max_steps: int = 500) -> int:
    """
    Move arm to safe candle (straight-up) position.
    Returns number of steps taken.
    """
    print("Tucking arm to safe candle position...")
    
    arm_q = get_arm_joint_states(robot_id, robot_config)
    initial_error = np.linalg.norm(arm_controller.target_configuration - arm_q)
    print(f"Initial arm error from candle: {initial_error:.4f} rad")
    
    steps = 0
    while steps < max_steps and not arm_controller.has_arrived(arm_q, threshold=threshold):
        action = np.zeros(robot_config.TOTAL_DOF)
        arm_vels = arm_controller.get_target_velocities(arm_q, target_pos_3d=None)
        action[robot_config.ARM_SLICE] = arm_vels
        
        env.step(action)
        arm_q = get_arm_joint_states(robot_id, robot_config)
        steps += 1
    
    final_error = np.linalg.norm(arm_controller.target_configuration - arm_q)
    print(f"Arm tucked after {steps} steps! Final error: {final_error:.4f} rad\n")
    
    return steps


def setup_environment(render: bool = False, random_obstacles: bool = False):
    """Initialize environment, robot, and obstacles."""
    model = GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel")
    
    # Create environment
    env = UrdfEnv(
        dt=0.01,
        robots=[model],
        render=render,
        num_sub_steps=300
    )
    
    # Create obstacle manager and populate environment
    playground = PlaygroundEnv(
        env=env,
        end_pos=(9.0, 4.5, 0.0),
        robot_radius=PandaConfig.BASE_RADIUS,
        random=random_obstacles,
        obstacle_count=2
    )
    
    # Store playground reference in env for later access
    env._playground = playground
    
    chair_marker = playground.chair_marker
    closet_marker = playground.closet_marker

    obstacles_2d = playground.get_2d_obstacles()
    ob = env.reset()
    
    # NOTE: Accessing private attributes as urdfenvs doesn't provide public API
    # for robot ID retrieval. This is a library limitation.
    robot_id = env._robots[0]._robot

    goals = playground.get_graspable_goals()
    print(f"\nFound {len(goals)} graspable objects:")
    for goal in goals:
        print(f"  - {goal['name']} at {goal['position']}")
    
    return env, obstacles_2d, ob, robot_id, goals

def setup_controllers(robot_id: int, obstacles_2d: list, mission: MissionConfig, robot_config):
    """Initialize path planner, follower, and arm controllers."""
    # Path planning for mobile base
    planner = RRTPlanner(
        obstacles=obstacles_2d,
        step_size=0.15, 
        max_iterations=2000,
        bounds=(-10.0, 10.0, -10.0, 10.0),
        robot_radius=PandaConfig.BASE_RADIUS,
        goal_sample_rate=0.10
    )
    
    follower = PathFollower(
        forward_velocity=mission.forward_velocity,
        waypoint_threshold=0.1
    )
    
    # Arm controllers with injected robot config
    mppi_ctrl = MppiArmController(robot_id=robot_id, robot_config=robot_config)
    safe_ctrl = ArmController(robot_id=robot_id, robot_config=robot_config, kp=15.0, max_vel=1.0)
    
    return planner, follower, mppi_ctrl, safe_ctrl


# --- Main Execution ---
def run_mobile_reacher(n_steps: int = 10000, render: bool = False, 
                       random_obstacles: bool = False):
    """
    State machine for mobile base navigation + arm reaching with multiple missions.
    
    States: TUCK -> DRIVE -> REACH -> (next mission)
    
    Args:
        n_steps: Maximum number of simulation steps
        render: Whether to render the simulation
        random_obstacles: Use random obstacle generation
    """
    # Configuration
    robot_config = PandaConfig()
    
    # Setup environment
    env, obstacles_2d, ob, robot_id, goals = setup_environment(render, random_obstacles)
    
    start_pos, _ = get_state_from_observation(ob, robot_config)
    mission_planner = MissionPlanner(robot_radius=PandaConfig.BASE_RADIUS, obstacles_2d=obstacles_2d)
    missions = mission_planner.plan_missions(goals, robot_start_pos=start_pos)
    
    print(f"\nGenerated {len(missions)} missions:")
    for i, mission in enumerate(missions):
        print(f"  Mission {i+1}: Base goal {mission.base_goal_2d}, Arm goal {mission.arm_goal_3d}")
    
    # Optional: Visualize obstacles and goals before planning
    if render:
        visualizer = PathVisualizer(obstacles_2d, missions[0].base_goal_2d)
        goal_positions = [m.base_goal_2d for m in missions]
        visualizer.show_obstacles_only(goals=goal_positions, current_pos=start_pos, 
                                       title="Environment with Goals")

    # Setup controllers
    planner, follower, mppi_ctrl, safe_ctrl = setup_controllers(
        robot_id, obstacles_2d, missions[0], robot_config
    )
    
    # Initialize state machine
    state_machine = MissionStateMachine(
        missions=missions,
        robot_config=robot_config,
        env=env,
        robot_id=robot_id,
        planner=planner,
        follower=follower,
        mppi_ctrl=mppi_ctrl,
        safe_ctrl=safe_ctrl,
        obstacles_2d=obstacles_2d,
        render=render
    )
    
    # Setup first mission
    print(f"\nStarting mission 1/{len(missions)}")
    print(f"{'='*60}\n")
    mission = missions[0]
    state_machine.target_marker_id = create_target_marker(mission.arm_goal_3d)
    
    # Plan initial path
    state_machine.path = planner.plan_path(start_pos, mission.base_goal_2d)
    print(f"Path is: {start_pos}, to {mission.base_goal_2d} waypoints\n")

    print(f"Path planned: {len(state_machine.path)} waypoints\n")
    
    if render:
        visualizer = PathVisualizer(obstacles_2d, mission.base_goal_2d)
        visualizer.show(state_machine.path, current_pos=start_pos, title="Mission 1 Path")
    
    # Main execution loop
    history = []
    
    for step in range(n_steps):
        action = state_machine.compute_action(ob, step)
        
        # Check if we should advance to next mission
        if action is None:
            has_more_missions = state_machine.advance_to_next_mission(ob)
            if not has_more_missions:
                break
            continue
        
        ob, _, _, _, info = env.step(action)
        history.append(ob)
        
        # Check for collisions
        if "Collision" in info:
            print(f"\nCollision at step {step}: {info['Collision']}")
    
    env.close()
    print(f"\nSimulation completed after {step + 1} steps")
    return history


if __name__ == "__main__":
    run_mobile_reacher(render=True, random_obstacles=False)
