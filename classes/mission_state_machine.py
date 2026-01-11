import numpy as np
import pybullet as p
from enum import Enum, auto
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv


class State(Enum):
    """State machine states."""

    TUCK = auto()
    DRIVE = auto()
    REACH = auto()


class MissionStateMachine:
    """State machine for executing multiple navigation and reaching missions."""

    def __init__(self, robot_config, scenario_config):
        """
        Initialize state machine with configuration.

        Args:
            robot_config: RobotConfig instance with all robot and controller settings
            scenario_config: ScenarioConfig instance with environment and simulation settings
        """
        self.robot_config = robot_config
        self.scenario_config = scenario_config

        # Setup environment (internal)
        self._setup_environment()

        # Get initial state
        self.ob = self.env.reset()

        # Get robot ID after reset (when _robot attribute is created)
        self.robot_id = self.env._robots[0]._robot

        start_pos, _, _ = self._get_state_from_observation(self.ob)

        # Plan missions
        self._plan_missions(start_pos)

        # Setup controllers (internal)
        self._setup_controllers()

        # Initialize state machine
        self.current_mission_idx = 0
        self.state = State.TUCK
        self.tuck_steps = 0
        self.reach_steps = 0

        # Setup first mission
        self._setup_first_mission(start_pos)

    def _setup_environment(self):
        """Initialize environment, robot, and obstacles."""
        from .playground import PlaygroundEnv

        model = GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel")

        # Create environment
        self.env = UrdfEnv(
            dt=self.scenario_config.dt,
            robots=[model],
            render=self.scenario_config.render,
            num_sub_steps=self.scenario_config.num_sub_steps,
        )

        # Create obstacle manager and populate environment
        playground = PlaygroundEnv(
            env=self.env,
            end_pos=self.scenario_config.end_pos,
            robot_radius=self.robot_config.base_radius,
            type=self.scenario_config.playground_type.value,
        )

        self.obstacles_2d = playground.get_2d_obstacles()

        self.goals = playground.get_graspable_goals()
        print(f"\nFound {len(self.goals)} graspable objects:")
        for goal in self.goals:
            print(f"  - {goal['name']} at {goal['position']}")

    def _plan_missions(self, start_pos):
        """Plan all missions from goals."""
        from .mission_planner import MissionPlanner

        mission_planner = MissionPlanner(
            robot_radius=self.robot_config.base_radius, obstacles_2d=self.obstacles_2d
        )
        self.missions = mission_planner.plan_missions(self.goals, robot_start_pos=start_pos)

        print(f"\nGenerated {len(self.missions)} missions:")
        for i, mission in enumerate(self.missions):
            print(
                f"  Mission {i + 1}: Base goal {mission.base_goal_2d}, Arm goal {mission.arm_goal_3d}"
            )

        # Show initial environment if configured
        if self.scenario_config.show_initial_environment:
            from .path_visualizer import PathVisualizer

            visualizer = PathVisualizer(self.obstacles_2d, self.missions[0].base_goal_2d)
            goal_positions = [m.base_goal_2d for m in self.missions]
            visualizer.show_obstacles_only(
                goals=goal_positions, current_pos=start_pos, title="Environment with Goals"
            )

    def _setup_controllers(self):
        """Initialize controllers based on configuration."""
        from .path_follower import PathFollower
        from .arm_controller import ArmController

        # Select and initialize the correct RRT planner
        planner_map = {
            "simple": "rrt_simple",
            "smooth": "rrt_smooth",
            "star": "rtt_star",
        }

        planner_module_name = planner_map[self.robot_config.planner.version.value]
        planner_module = __import__(f"classes.{planner_module_name}", fromlist=["RRTPlanner"])
        planner_cls = planner_module.RRTPlanner

        cfg = self.robot_config.planner
        self.planner = planner_cls(
            obstacles=self.obstacles_2d,
            step_size=cfg.step_size,
            max_iterations=cfg.max_iterations,
            goal_threshold=cfg.goal_threshold,
            bounds=cfg.bounds,
            robot_radius=self.robot_config.base_radius,  # Use consistent base_radius
            goal_sample_rate=cfg.goal_sample_rate,
        )

        # Initialize path follower
        self.follower = PathFollower(
            forward_velocity=self.robot_config.path_follower.forward_velocity,
            waypoint_threshold=self.robot_config.path_follower.waypoint_threshold,
        )

        # Select and initialize the correct MPPI controller
        mppi_map = {
            "basic": ("mppi_arm_controller_basic", "MppiArmControllerBasic"),
            "advanced": ("mppi_arm_controller", "MppiArmController"),
        }

        mppi_module_name, mppi_class_name = mppi_map[self.robot_config.mppi.version.value]
        mppi_module = __import__(f"classes.{mppi_module_name}", fromlist=[mppi_class_name])
        mppi_cls = getattr(mppi_module, mppi_class_name)

        cfg = self.robot_config.mppi

        # Both MPPI versions share common parameters
        common_params = {
            "robot_id": self.robot_id,
            "robot_config": self.robot_config,
            "dt": cfg.dt,
            "horizon": cfg.horizon,
            "n_samples": cfg.n_samples,
            "lambda_": cfg.lambda_,
            "sigma": cfg.sigma,
            "dist_weight": cfg.dist_weight,
            "collision_cost": cfg.collision_cost,
            "jerk_weight": cfg.jerk_weight,
        }
        # Advanced version has additional parameters
        if cfg.version.value == "advanced":
            common_params.update(
                {
                    "terminal_dist_weight": cfg.terminal_dist_weight,
                    "exp_decay_rate": cfg.exp_decay_rate,
                }
            )

        self.mppi_ctrl = mppi_cls(**common_params)

        # Safe arm controller for tucking
        self.safe_ctrl = ArmController(
            robot_id=self.robot_id,
            robot_config=self.robot_config,
            dt=self.scenario_config.dt,
            kp=self.robot_config.arm_controller.kp,
            max_vel=self.robot_config.arm_controller.max_vel,
        )

    def _setup_first_mission(self, start_pos):
        """Setup the first mission with path planning and visualization."""
        print(f"\nStarting mission 1/{len(self.missions)}")
        print(f"{'=' * 60}\n")

        mission = self.missions[0]
        self.target_marker_id = self.create_target_marker(
            mission.arm_goal_3d, size=self.robot_config.goal_reach_threshold
        )

        # Plan initial path
        print("mission_base_goal_2d:", mission.base_goal_2d)
        self.path = self.planner.plan_path(start_pos, mission.base_goal_2d)

        print(f"Path is: {start_pos}, to {mission.base_goal_2d} waypoints\n")
        print("machine_path", self.path)
        print(f"Path planned: {len(self.path)} waypoints\n")

        if self.scenario_config.render and self.scenario_config.show_path_plots:
            from .path_visualizer import PathVisualizer

            visualizer = PathVisualizer(self.obstacles_2d, mission.base_goal_2d)
            visualizer.show(self.path, current_pos=start_pos, title="Mission 1 Path")

    def run(self):
        """Main execution loop. Returns observation history."""
        history = []

        for step in range(self.scenario_config.n_steps):
            action = self.compute_action(self.ob, step)

            if action is None:
                has_more = self.advance_to_next_mission(self.ob)
                if not has_more:
                    break
                continue

            self.ob, _, _, _, info = self.env.step(action)
            history.append(self.ob)

            if "Collision" in info:
                print(f"\nCollision at step {step}: {info['Collision']}")

        self.env.close()
        print(f"\nSimulation completed after {step + 1} steps")
        return history

    def get_current_mission(self):
        """Get the current mission configuration."""
        return self.missions[self.current_mission_idx]

    def advance_to_next_mission(self, ob):
        """Move to next mission. Returns True if more missions available."""
        self.current_mission_idx += 1

        if self.current_mission_idx >= len(self.missions):
            print("\nAll missions completed!")
            return False

        print(f"\n{'=' * 60}")
        print(f"Starting mission {self.current_mission_idx + 1}/{len(self.missions)}")
        print(f"{'=' * 60}\n")

        # Remove old target marker
        if self.target_marker_id is not None:
            p.removeBody(self.target_marker_id)

        # Setup new mission
        mission = self.get_current_mission()
        self.target_marker_id = self._create_target_marker(
            mission.arm_goal_3d, size=self.robot_config.goal_reach_threshold * 2
        )

        # Plan path to new destination
        base_xy, _, _ = self._get_state_from_observation(ob)
        print(f"Path is: {base_xy}, to {mission.base_goal_2d} waypoints\n")
        self.path = self.planner.plan_path(base_xy, mission.base_goal_2d)
        print(f"Path planned: {len(self.path)} waypoints\n")

        if self.scenario_config.render and self.scenario_config.show_path_plots:
            from .path_visualizer import PathVisualizer

            visualizer = PathVisualizer(self.obstacles_2d, mission.base_goal_2d)
            visualizer.show(
                self.path, current_pos=base_xy, title=f"Mission {self.current_mission_idx + 1} Path"
            )

        # Transition to TUCK state for new mission
        self.state = State.TUCK
        self.tuck_steps = 0
        self.reach_steps = 0

        return True

    def compute_action(self, ob, step):
        """Compute action based on current state."""
        base_xy, base_full, arm_q = self._get_state_from_observation(ob)
        action = np.zeros(self.robot_config.total_dof)
        mission = self.get_current_mission()

        if self.state == State.TUCK:
            if self.tuck_steps == 0:
                print("[TUCK] Tucking arm to safe candle position...")

            arm_vels = self.safe_ctrl.get_target_velocities(arm_q, target_pos_3d=None)
            action[self.robot_config.arm_slice] = arm_vels
            self.tuck_steps += 1

            # Only check if arm is tucked after warm-up period
            if self.tuck_steps >= self.robot_config.tuck_warmup_steps:
                if self.safe_ctrl.has_arrived(arm_q, threshold=0.1):
                    print("[TUCK] Arm tucked successfully\n")
                    self.state = State.DRIVE
                    self.tuck_steps = 0

        elif self.state == State.DRIVE:
            # Navigate to base goal
            dist_to_goal = np.linalg.norm(base_xy - mission.base_goal_2d)

            if dist_to_goal > self.robot_config.switch_distance:
                # Get base velocities from path follower
                base_vel = self.follower.follow(
                    base_xy, self.path, self.robot_config.total_dof, final_target=mission.arm_goal_3d
                )
                action[self.robot_config.base_slice] = base_vel[self.robot_config.base_slice]
                
                # Always orient towards the arm goal
                arm_goal_2d = np.array(mission.arm_goal_3d[:2])
                direction_to_goal = arm_goal_2d - base_xy
                desired_angle = np.arctan2(direction_to_goal[1], direction_to_goal[0])
                current_angle = base_full[2]
                angle_error = desired_angle - current_angle
                # Normalize angle to [-pi, pi]
                angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
                action[2] = angle_error * 2.0  # Angular velocity proportional to error
            else:
                print(f"[DRIVE] Reached destination at step {step}\n")
                self.state = State.REACH
                self.reach_steps = 0

        elif self.state == State.REACH:
            if self.reach_steps == 0:
                print(f"[REACH] Starting arm reach to target {mission.arm_goal_3d}")

            # Base stays still during reach
            action[self.robot_config.base_slice] = 0.0

            # MPPI arm control with base position
            arm_vels = self.mppi_ctrl.compute_action(
                arm_q,
                mission.arm_goal_3d,
                target_body_id=self.target_marker_id,
                # base_pos=base_full,
            )
            action[self.robot_config.arm_slice] = arm_vels
            self.reach_steps += 1

            # ---------------- EXIT CONDITIONS ---------------- #

            # Inline end-effector distance check
            ee_pos = np.array(p.getLinkState(self.robot_id, self.robot_config.ee_link_index)[0])
            dist = np.linalg.norm(ee_pos - np.array(mission.arm_goal_3d))

            # Success
            if dist < self.robot_config.goal_reach_threshold:
                print("[REACH] Target reached")

                # Clean up MPPI + PyBullet state
                if hasattr(self.mppi_ctrl, "cleanup_after_goal"):
                    self.mppi_ctrl.cleanup_after_goal()

                return None  # advance to next mission

        # Always keep gripper still
        action[self.robot_config.gripper_slice] = 0.0
        return action

    @staticmethod
    def get_state_from_observation(obs, robot_config):
        """Extract base (x,y,theta) and arm joints from observation."""
        if isinstance(obs, dict) and robot_config.robot_name in obs:
            state = obs[robot_config.robot_name]["joint_state"]["position"]
            base_xy = np.array(state[:2])
            base_full = np.array(state[:3])  # x, y, theta for FK
            arm_q = np.array(state[robot_config.arm_obs_slice])
            return base_xy, base_full, arm_q
        return np.zeros(2), np.zeros(3), np.zeros(7)

    def _get_state_from_observation(self, obs):
        """Internal alias for get_state_from_observation."""
        return self.get_state_from_observation(obs, self.robot_config)

    def get_arm_joint_states(self):
        """Extract current arm joint positions from PyBullet."""
        return np.array(
            [p.getJointState(self.robot_id, idx)[0] for idx in self.robot_config.arm_joint_indices]
        )

    @staticmethod
    def create_target_marker(position, size=0.05, color=(1.0, 0.0, 1.0, 0.6)):
        """Create a target marker in PyBullet (visual only, no collision)."""
        half_size = size / 2.0

        # Create only visual shape (no collision)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX, halfExtents=[half_size, half_size, half_size], rgbaColor=color
        )

        return p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=-1,  # -1 means no collision shape
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

    def _create_target_marker(self, position, size=0.05, color=(1.0, 0.0, 1.0, 0.6)):
        """Internal alias for create_target_marker."""
        return self.create_target_marker(position, size, color)

    def remove_collided_obstacle(self):
        """
        Remove the obstacle that the robot collided with.

        Returns:
            True if an obstacle was removed
        """
        num_bodies = p.getNumBodies()

        for body_id in range(num_bodies):
            if body_id == self.robot_id:
                continue

            # Check if this body is in contact with robot
            contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=body_id)
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
