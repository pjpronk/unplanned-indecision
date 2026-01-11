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

    def __init__(
        self,
        missions: list,
        robot_config,
        env,
        robot_id,
        planner,
        follower,
        mppi_ctrl,
        safe_ctrl,
        obstacles_2d,
        render: bool = False,
    ):
        self.missions = missions
        self.robot_config = robot_config
        self.env = env
        self.robot_id = robot_id
        self.planner = planner
        self.follower = follower
        self.mppi_ctrl = mppi_ctrl
        self.safe_ctrl = safe_ctrl
        self.obstacles_2d = obstacles_2d
        self.render = render

        # State machine variables
        self.current_mission_idx = 0
        self.state = State.TUCK
        self.tuck_steps = 0
        self.tuck_warmup_steps = 20  # Steps to wait before checking if arm is tucked

        # Current mission data
        self.path = None
        self.target_marker_id = None

    def get_current_mission(self):
        """Get the current mission configuration."""
        return self.missions[self.current_mission_idx]

    def advance_to_next_mission(self, ob) -> bool:
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
        from .path_visualizer import PathVisualizer

        self.target_marker_id = self._create_target_marker(mission.arm_goal_3d)

        # Plan path to new destination
        base_xy, _ = self._get_state_from_observation(ob)
        print(f"Path is: {base_xy}, to {mission.base_goal_2d} waypoints\n")
        self.path = self.planner.plan_path(base_xy, mission.base_goal_2d)
        print(f"Path planned: {len(self.path)} waypoints\n")

        if self.render:
            visualizer = PathVisualizer(self.obstacles_2d, mission.base_goal_2d)
            visualizer.show(
                self.path, current_pos=base_xy, title=f"Mission {self.current_mission_idx + 1} Path"
            )

        # Transition to TUCK state for new mission
        self.state = State.TUCK
        self.tuck_steps = 0
        self.reach_steps = 0

        return True

    def compute_action(self, ob, step: int) -> np.ndarray:
        """Compute action based on current state."""
        base_xy, arm_q = self._get_state_from_observation(ob)
        action = np.zeros(self.robot_config.TOTAL_DOF)
        mission = self.get_current_mission()

        if self.state == State.TUCK:
            # Tuck arm to safe position
            if self.tuck_steps == 0:
                print("[TUCK] Tucking arm to safe candle position...")

            arm_vels = self.safe_ctrl.get_target_velocities(arm_q, target_pos_3d=None)
            action[self.robot_config.ARM_SLICE] = arm_vels
            self.tuck_steps += 1

            # Only check if arm is tucked after warm-up period
            if self.tuck_steps >= self.tuck_warmup_steps:
                if self.safe_ctrl.has_arrived(arm_q, threshold=0.1):
                    print("[TUCK] Arm tucked successfully\n")
                    self.state = State.DRIVE
                    self.tuck_steps = 0

        elif self.state == State.DRIVE:
            # Navigate to base goal
            dist_to_goal = np.linalg.norm(base_xy - mission.base_goal_2d)

            if dist_to_goal > mission.switch_distance:
                base_vel = self.follower.follow(base_xy, self.path, self.robot_config.TOTAL_DOF)
                action[self.robot_config.BASE_SLICE] = base_vel[self.robot_config.BASE_SLICE]
            else:
                print(f"[DRIVE] Reached destination at step {step}\n")
                self.state = State.REACH
                self.reach_steps = 0

        elif self.state == State.REACH:
            # Use MPPI to reach target
            if self.reach_steps == 0:
                print(f"[REACH] Starting arm reach to target {mission.arm_goal_3d}")

            action[self.robot_config.BASE_SLICE] = 0.0
            arm_vels = self.mppi_ctrl.compute_action(
                arm_q, mission.arm_goal_3d, target_body_id=self.target_marker_id
            )
            action[self.robot_config.ARM_SLICE] = arm_vels
        # Always keep gripper still
        action[self.robot_config.GRIPPER_SLICE] = 0.0
        return action

    @staticmethod
    def get_state_from_observation(obs: dict, robot_config) -> tuple:
        """Extract base (x,y) and arm joints from observation."""
        if isinstance(obs, dict) and robot_config.ROBOT_NAME in obs:
            state = obs[robot_config.ROBOT_NAME]["joint_state"]["position"]
            base_xy = np.array(state[:2])
            arm_q = np.array(state[robot_config.ARM_OBS_SLICE])
            return base_xy, arm_q
        return np.zeros(2), np.zeros(7)

    def _get_state_from_observation(self, obs: dict) -> tuple:
        """Internal alias for get_state_from_observation."""
        return self.get_state_from_observation(obs, self.robot_config)

    def get_arm_joint_states(self) -> np.ndarray:
        """Extract current arm joint positions from PyBullet."""
        return np.array(
            [p.getJointState(self.robot_id, idx)[0] for idx in self.robot_config.ARM_JOINT_INDICES]
        )

    @staticmethod
    def create_target_marker(
        position: list, size: float = 0.05, color: tuple = (1.0, 0.0, 1.0, 0.6)
    ) -> int:
        """Create a target marker in PyBullet with collision shape."""
        half_size = size / 2.0

        # Create both visual and collision shapes
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX, halfExtents=[half_size, half_size, half_size], rgbaColor=color
        )

        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=[half_size, half_size, half_size]
        )

        return p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

    def _create_target_marker(
        self, position: list, size: float = 0.05, color: tuple = (1.0, 0.0, 1.0, 0.6)
    ) -> int:
        """Internal alias for create_target_marker."""
        return self.create_target_marker(position, size, color)

    def remove_collided_obstacle(self) -> bool:
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

    @staticmethod
    def setup_environment(render: bool = False, random_obstacles: bool = False):
        """Initialize environment, robot, and obstacles."""
        from .playground import PlaygroundEnv
        from .robot_config import PandaConfig

        model = GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel")

        # Create environment
        env = UrdfEnv(dt=0.01, robots=[model], render=render, num_sub_steps=300)

        # Create obstacle manager and populate environment
        playground = PlaygroundEnv(
            env=env,
            end_pos=(9.0, 4.5, 0.0),
            robot_radius=PandaConfig.BASE_RADIUS,
            random=random_obstacles,
            obstacle_count=2,
        )

        # Store playground reference in env for later access
        env._playground = playground

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

    @staticmethod
    def setup_controllers(robot_id: int, obstacles_2d: list, mission, robot_config):
        """Initialize path planner, follower, and arm controllers."""
        from .rrt_smooth import RRTPlanner
        from .path_follower import PathFollower
        from .arm_controller import ArmController
        from .mppi_arm_controller_basic import MppiArmControllerBasic

        # Path planning for mobile base
        planner = RRTPlanner(
            obstacles=obstacles_2d,
            step_size=0.15,
            max_iterations=2000,
            bounds=(-10.0, 10.0, -10.0, 10.0),
            robot_radius=0.25,
            goal_sample_rate=0.10,
        )

        follower = PathFollower(forward_velocity=mission.forward_velocity, waypoint_threshold=0.1)

        # Arm controllers with injected robot config
        mppi_ctrl = MppiArmControllerBasic(robot_id=robot_id, robot_config=robot_config)
        safe_ctrl = ArmController(
            robot_id=robot_id, robot_config=robot_config, kp=15.0, max_vel=1.0
        )

        return planner, follower, mppi_ctrl, safe_ctrl
