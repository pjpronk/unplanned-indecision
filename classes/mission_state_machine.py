import numpy as np
import pybullet as p
from enum import Enum, auto


class State(Enum):
    """State machine states."""
    TUCK = auto()
    DRIVE = auto()
    REACH = auto()


class MissionStateMachine:
    """State machine for executing multiple navigation and reaching missions."""
    
    def __init__(self, missions: list, robot_config, env, robot_id, 
                 planner, follower, mppi_ctrl, safe_ctrl, obstacles_2d, render: bool = False):
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
        
        print(f"\n{'='*60}")
        print(f"Starting mission {self.current_mission_idx + 1}/{len(self.missions)}")
        print(f"{'='*60}\n")
        
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
            visualizer.show(self.path, current_pos=base_xy, 
                          title=f"Mission {self.current_mission_idx + 1} Path")
        
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
                print(f"[TUCK] Tucking arm to safe candle position...")
            
            arm_vels = self.safe_ctrl.get_target_velocities(arm_q, target_pos_3d=None)
            action[self.robot_config.ARM_SLICE] = arm_vels
            self.tuck_steps += 1
            
            # Only check if arm is tucked after warm-up period
            if self.tuck_steps >= self.tuck_warmup_steps:
                if self.safe_ctrl.has_arrived(arm_q, threshold=0.1):
                    print(f"[TUCK] Arm tucked successfully\n")
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
                arm_q,
                mission.arm_goal_3d,
                target_body_id=self.target_marker_id
            )
            action[self.robot_config.ARM_SLICE] = arm_vels
        # Always keep gripper still
        action[self.robot_config.GRIPPER_SLICE] = 0.0
        return action
    
    def _get_state_from_observation(self, obs: dict) -> tuple:
        """Extract base (x,y) and arm joints from observation."""
        if isinstance(obs, dict) and self.robot_config.ROBOT_NAME in obs:
            state = obs[self.robot_config.ROBOT_NAME]["joint_state"]["position"]
            base_xy = np.array(state[:2])
            arm_q = np.array(state[self.robot_config.ARM_OBS_SLICE])
            return base_xy, arm_q
        return np.zeros(2), np.zeros(7)
    
    @staticmethod
    def _create_target_marker(position: list, size: float = 0.05, 
                              color: tuple = (1.0, 0.0, 1.0, 0.6)) -> int:
        """Create a target marker in PyBullet with collision shape."""
        half_size = size / 2.0
        
        # Create both visual and collision shapes
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size, half_size],
            rgbaColor=color
        )
        
        return p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1]
        )

