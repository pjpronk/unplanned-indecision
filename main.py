import numpy as np
import pybullet as p

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from classes import RRTPlanner, PathFollower, ObstacleManager, PathVisualizer, ArmController, MppiArmController

def run_mobile_reacher(n_steps=10_000, render=False, goal=True, obstacles=True):
    """
    Mobile base drives along an RRT path to a 2D goal. Once close enough, it stops and
    uses MPPI to move the arm end-effector to a 3D target while avoiding obstacles.
    """

    # --- Targets / thresholds ---
    base_goal_2d = np.array([2.0, -1.5])
    arm_goal_3d = [2.5, -1.5, 0.2]      # visual marker location (and reach target)
    switch_dist = 0.15                  # switch from DRIVE to REACH when within this
    fwd_vel = 1.5
    action_size = 12                    # [vx, vy, w, j1..j7, g1, g2]

    # --- Environment & robot ---
    model = GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel")
    env = UrdfEnv(dt=0.01, robots=[model], render=render, num_sub_steps=200)

    # --- Obstacles (2D for RRT, 3D for MPPI), plus a visual-only target cube ---
    obst_manager = ObstacleManager()
    obstacles_2d = obst_manager.get_2d_obstacles()
    if obstacles:
        obst_manager.add_to_urdf_env(env)

    # --- Planning / visualization / base controller ---
    planner = RRTPlanner(obstacles=obstacles_2d)
    follower = PathFollower(forward_velocity=fwd_vel, waypoint_threshold=0.1)
    visualizer = PathVisualizer(obstacles_2d, base_goal_2d)

    # --- Reset (spawns robot + obstacles in PyBullet) ---
    ob = env.reset()

    # Spawn visual-only target marker (no collision shape)
    target_body_id = obst_manager.add_target_visual(arm_goal_3d)
    print(f"Visual-only target marker body ID: {target_body_id}")

    # Robot ID (PyBullet body unique id) for MPPI collision queries
    robot_obj = env._robots[0]
    robot_id = robot_obj._robot
    print(f"Robot successfully found with ID: {robot_id}")

    # --- Arm MPPI controller (dt here is MPPI internal rollout dt/horizon step) ---
    arm_ctrl = MppiArmController(robot_id=robot_id, dt=0.05)

    # --- Initial base position (from observation) ---
    start_pos = np.array([0.0, 0.0])
    if isinstance(ob, dict) and "robot_0" in ob:
        start_pos = np.array(ob["robot_0"]["joint_state"]["position"][:2])

    # --- Plan path once ---
    path = planner.plan_path(start_pos, base_goal_2d)
    print(f"Path found: {len(path)} waypoints")
    if render:
        visualizer.show(path, current_pos=start_pos, title="Initial Plan")

    # --- Helpers ---
    def get_state(obs):
        """Extract base (x,y) and 7 arm joints from observation."""
        if isinstance(obs, dict) and "robot_0" in obs:
            state = obs["robot_0"]["joint_state"]["position"]
            base_xy = np.array(state[:2])
            arm_q = np.array(state[3:10])  # j1..j7
            return base_xy, arm_q
        return np.zeros(2), np.zeros(7)

    # --- Main loop ---
    history = []
    for step in range(n_steps):
        base_xy, arm_q = get_state(ob)
        dist_to_goal = np.linalg.norm(base_xy - base_goal_2d)

        action = np.zeros(action_size)

        if dist_to_goal > switch_dist:
            # DRIVE: follow planned path with base velocities; keep arm still.
            base_vel = follower.follow(base_xy, path, action_size)
            action[:] = base_vel
            action[3:10] = 0.0
        else:
            # REACH: stop base, use MPPI arm velocities toward 3D goal.
            action[0:3] = 0.0
            arm_vels = arm_ctrl.compute_action(
                arm_q,
                arm_goal_3d,
                target_body_id=target_body_id,  # visual-only marker; safe to ignore if contacted
            )
            action[3:10] = arm_vels

        ob, _, _, _, info = env.step(action)
        history.append(ob)

        if "Collision" in info:
            print(f"\nCRITICAL: Collision at step {step}!")
            print(f"Details: {info['Collision']}")
            break

    env.close()
    return history


if __name__ == "__main__":
    run_mobile_reacher(render=True)
