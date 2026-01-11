import platform

import matplotlib

# Override for Linux
if platform.system() == "Linux":
    matplotlib.use("TkAgg", force=True)


from classes import PathVisualizer, PandaConfig, MissionStateMachine, MissionPlanner


# --- Main Execution ---
def run_mobile_reacher(n_steps: int = 10000, render: bool = True, random_obstacles: bool = False):
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
    env, obstacles_2d, ob, robot_id, goals = MissionStateMachine.setup_environment(
        render, random_obstacles
    )

    # Get initial state
    start_pos, _ = MissionStateMachine.get_state_from_observation(ob, robot_config)

    # Plan missions
    mission_planner = MissionPlanner(
        robot_radius=PandaConfig.BASE_RADIUS, obstacles_2d=obstacles_2d
    )
    missions = mission_planner.plan_missions(goals, robot_start_pos=start_pos)

    visualizer = PathVisualizer(obstacles_2d, missions[0].base_goal_2d)
    goal_positions = [m.base_goal_2d for m in missions]
    visualizer.show_obstacles_only(
        goals=goal_positions, current_pos=start_pos, title="Environment with Goals"
    )

    print(f"\nGenerated {len(missions)} missions:")
    for i, mission in enumerate(missions):
        print(
            f"  Mission {i + 1}: Base goal {mission.base_goal_2d}, Arm goal {mission.arm_goal_3d}"
        )

    # Setup controllers
    planner, follower, mppi_ctrl, safe_ctrl = MissionStateMachine.setup_controllers(
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
        render=render,
    )

    # Setup first mission
    print(f"\nStarting mission 1/{len(missions)}")
    print(f"{'=' * 60}\n")
    mission = missions[0]
    state_machine.target_marker_id = MissionStateMachine.create_target_marker(mission.arm_goal_3d)

    # Plan initial path
    print("mission_base_goal_2d:", mission.base_goal_2d)
    state_machine.path = planner.plan_path(start_pos, mission.base_goal_2d)

    print(f"Path is: {start_pos}, to {mission.base_goal_2d} waypoints\n")
    print("machine_path", state_machine.path)
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
