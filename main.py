import platform
from enum import Enum
from dataclasses import dataclass, field
import numpy as np

import matplotlib

# Override for Linux
if platform.system() == "Linux":
    matplotlib.use("TkAgg", force=True)


from classes import MissionStateMachine


class MPPIVersion(Enum):
    """Available MPPI controller implementations"""

    BASIC = "basic"
    ADVANCED = "advanced"


class PlannerVersion(Enum):
    """Available RRT planner implementations"""

    SIMPLE = "simple"
    SMOOTH = "smooth"
    STAR = "star"


class PlaygroundType(Enum):
    """Predefined environment layouts"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    QUANTUM_RRT = "quantum_RRT"
    MPPI_TEST = "mppi_test"


@dataclass
class MPPIConfig:
    """MPPI controller configuration"""

    version: MPPIVersion = MPPIVersion.BASIC
    dt: float = 0.05
    horizon: int = 30
    n_samples: int = 30
    lambda_: float = 0.001
    sigma: float = 0.3
    dist_weight: float = 10.0
    collision_cost: float = 10_000.0
    jerk_weight: float = 1.0
    terminal_dist_weight: float = 500.0
    exp_decay_rate: float = 5.0
    sigma_min_factor: float = 0.3
    sigma_max_factor: float = 1.0
    adaptive_distance: float = 0.5


@dataclass
class ArmControllerConfig:
    """Safe arm controller (for tucking) configuration"""

    kp: float = 15.0
    max_vel: float = 1.0


@dataclass
class PlannerConfig:
    """RRT path planner configuration"""

    version: PlannerVersion = PlannerVersion.SMOOTH
    step_size: float = 0.15
    max_iterations: int = 2000
    goal_threshold: float = 0.1
    bounds: tuple = (-10.0, 10.0, -10.0, 10.0)
    goal_sample_rate: float = 0.10
    search_radius: float = 0.5
    smoothing_iterations: int = 100
    smoothing_step: float = 0.05


@dataclass
class PathFollowerConfig:
    """Path following controller configuration"""

    forward_velocity: float = 1.5
    waypoint_threshold: float = 0.1


@dataclass
class RobotConfig:
    """Robot-specific configuration"""

    # Robot identification
    robot_name: str = "robot_0"
    base_radius: float = 0.3

    # Joint indices in URDF
    base_joint_indices: list = field(default_factory=lambda: [0, 1, 2])
    arm_joint_indices: list = field(default_factory=lambda: [4, 5, 6, 7, 8, 9, 10])
    gripper_joint_indices: list = field(default_factory=lambda: [11, 12])

    # Link indices
    ee_link_index: int = 14
    base_link_index: int = -1

    # Controller configs
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    arm_controller: ArmControllerConfig = field(default_factory=ArmControllerConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    path_follower: PathFollowerConfig = field(default_factory=PathFollowerConfig)

    # Mission parameters
    switch_distance: float = 0.1
    tuck_warmup_steps: int = 20
    goal_reach_threshold: float = 0.2  # Distance threshold for reaching arm goal (meters)

    # Degrees of freedom
    base_dof: int = 3
    arm_dof: int = 7
    gripper_dof: int = 2
    total_dof: int = 12

    # Action vector slices
    base_slice: slice = field(default=slice(0, 3), init=False)
    arm_slice: slice = field(default=slice(3, 10), init=False)
    gripper_slice: slice = field(default=slice(10, 12), init=False)

    # Observation slices
    base_obs_slice: slice = field(default=slice(0, 3), init=False)
    arm_obs_slice: slice = field(default=slice(3, 10), init=False)

    # Panda arm joint limits (radians)
    arm_joint_limits_lower: np.ndarray = field(
        default_factory=lambda: np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        ),
        init=False,
    )
    arm_joint_limits_upper: np.ndarray = field(
        default_factory=lambda: np.array([2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973]),
        init=False,
    )

    # Candle configuration (arm positioned in the air, safe for navigation)
    candle_configuration: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.785, 0.0, -1.57, 0.0, 0.785, 0.785]), init=False
    )


@dataclass
class ScenarioConfig:
    """Environment and simulation scenario configuration"""

    name: str = "default"
    render: bool = True
    n_steps: int = 10000
    dt: float = 0.01
    num_sub_steps: int = 300
    playground_type: PlaygroundType = PlaygroundType.EASY
    end_pos: tuple = (9.0, 4.5, 0.0)
    show_path_plots: bool = True
    show_initial_environment: bool = True


class ConfigPresets:
    """Predefined configuration presets for common test scenarios"""

    @staticmethod
    def simple_test():
        """Simplest configuration: basic MPPI, simple RRT, easy environment"""
        robot = RobotConfig()
        robot.mppi = MPPIConfig()
        robot.mppi.version = MPPIVersion.BASIC
        robot.mppi.horizon = 10
        robot.mppi.n_samples = 10

        robot.planner = PlannerConfig()
        robot.planner.version = PlannerVersion.SIMPLE
        robot.planner.max_iterations = 1000

        scenario = ScenarioConfig()
        scenario.name = "simple_test"
        scenario.playground_type = PlaygroundType.EASY
        scenario.render = True
        scenario.show_path_plots = True

        return robot, scenario

    @staticmethod
    def advanced_smooth():
        """Advanced MPPI with smooth RRT for better performance"""
        robot = RobotConfig()
        robot.mppi = MPPIConfig()
        robot.mppi.version = MPPIVersion.ADVANCED
        robot.mppi.horizon = 30
        robot.mppi.n_samples = 50
        robot.mppi.dist_weight = 100.0
        robot.mppi.terminal_dist_weight = 500.0

        robot.planner = PlannerConfig()
        robot.planner.version = PlannerVersion.SIMPLE
        robot.planner.max_iterations = 2000
        robot.planner.smoothing_iterations = 100

        robot.path_follower = PathFollowerConfig()
        robot.path_follower.forward_velocity = 2.0

        scenario = ScenarioConfig()
        scenario.name = "advanced_smooth"
        scenario.playground_type = PlaygroundType.EASY
        scenario.render = True

        return robot, scenario

    @staticmethod
    def optimal_star():
        """Optimal path planning with RRT*"""
        robot = RobotConfig()
        robot.mppi = MPPIConfig()
        robot.mppi.version = MPPIVersion.ADVANCED
        robot.mppi.horizon = 30
        robot.mppi.n_samples = 100

        robot.planner = PlannerConfig()
        robot.planner.version = PlannerVersion.STAR
        robot.planner.max_iterations = 3000
        robot.planner.search_radius = 0.5

        scenario = ScenarioConfig()
        scenario.name = "optimal_star"
        scenario.playground_type = PlaygroundType.HARD
        scenario.render = True

        return robot, scenario

    @staticmethod
    def mppi_basic_test():
        """Test basic MPPI with simple obstacle - stationary robot, arm only"""
        robot = RobotConfig()
        robot.mppi = MPPIConfig()
        robot.mppi.version = MPPIVersion.BASIC
        robot.mppi.horizon = 20
        robot.mppi.n_samples = 50
        robot.mppi.dist_weight = 50.0
        robot.mppi.collision_cost = 100.0

        robot.planner = PlannerConfig()
        robot.planner.version = PlannerVersion.SIMPLE
        robot.planner.max_iterations = 1000

        scenario = ScenarioConfig()
        scenario.name = "mppi_basic_test"
        scenario.playground_type = PlaygroundType.MPPI_TEST
        scenario.render = True

        return robot, scenario

    @staticmethod
    def mppi_advanced_test():
        """Test advanced MPPI with simple obstacle - stationary robot, arm only"""
        robot = RobotConfig()
        robot.mppi = MPPIConfig()
        robot.mppi.version = MPPIVersion.ADVANCED
        robot.mppi.horizon = 25
        robot.mppi.n_samples = 80
        robot.mppi.dist_weight = 80.0
        robot.mppi.terminal_dist_weight = 200.0
        robot.mppi.collision_cost = 150.0

        robot.planner = PlannerConfig()
        robot.planner.version = PlannerVersion.SIMPLE
        robot.planner.max_iterations = 1000

        scenario = ScenarioConfig()
        scenario.name = "mppi_advanced_test"
        scenario.playground_type = PlaygroundType.MPPI_TEST
        scenario.render = True

        return robot, scenario


if __name__ == "__main__":
    robot_cfg, scenario_cfg = ConfigPresets.mppi_advanced_test()

    # Determine if stationary mode based on playground type
    stationary_mode = scenario_cfg.playground_type == PlaygroundType.MPPI_TEST

    print(f"\n{'-' * 60}")
    print(f"Running scenario: {scenario_cfg.name}")
    print(f"MPPI: {robot_cfg.mppi.version.value}")
    print(f"Planner: {robot_cfg.planner.version.value}")
    print(f"Environment: {scenario_cfg.playground_type.value}")
    print(f"Stationary Mode: {stationary_mode}")
    print(f"{'-' * 60}\n")

    state_machine = MissionStateMachine(robot_cfg, scenario_cfg, stationary_mode=stationary_mode)

    history = state_machine.run()
    print(f"\nCompleted with {len(history)} steps of history")
