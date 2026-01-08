import numpy as np
from dataclasses import dataclass


@dataclass
class MissionConfig:
    """Mission parameters for navigation and reaching."""
    base_goal_2d: np.ndarray
    arm_goal_3d: tuple
    switch_distance: float = 0.15
    forward_velocity: float = 1.5
    
    @classmethod
    def lego_pile_mission(cls):
        """Predefined mission to lego pile area."""
        return cls(
            base_goal_2d=np.array([6.0, 3.0]),
            arm_goal_3d=(6.3, 3.3, 0.8),
            switch_distance=0.3,
            forward_velocity=1.5
        )


class MissionPlanner:
    """
    Plans missions from environment goals.
    Computes safe base positions and generates MissionConfig objects.
    """
    
    def __init__(self, robot_radius: float = 0.3, obstacles_2d: list = None):
        """
        Initialize mission planner.
        
        Args:
            robot_radius: Robot base radius for safety calculations
            obstacles_2d: List of 2D obstacles for collision checking
        """
        self.robot_radius = robot_radius
        self.obstacles_2d = obstacles_2d or []
    
    def plan_missions(self, goals: list, robot_start_pos: np.ndarray = None) -> list:
        """
        Generate MissionConfig objects from goals.
        
        Args:
            goals: List of goal dicts from PlaygroundEnv.get_graspable_goals()
            robot_start_pos: Current robot position [x, y] for approach planning
        
        Returns:
            List of MissionConfig objects
        """
        missions = []
        
        for goal in goals:
            mission = self._plan_single_mission(goal, robot_start_pos)
            if mission:
                missions.append(mission)
        
        return missions
    
    def _plan_single_mission(self, goal: dict, robot_pos: np.ndarray = None) -> MissionConfig:
        """
        Plan a single mission from a goal.
        
        Args:
            goal: Goal dictionary with position, name, object_type, radius
            robot_pos: Current robot position for approach direction
        
        Returns:
            MissionConfig object
        """
        arm_goal_3d = tuple(goal['position'])
        goal_2d = goal['position'][:2]
        
        # Calculate safe base position
        safe_radius = self._get_safe_radius(goal)
        approach_direction = self._calculate_approach_direction(goal_2d, robot_pos)
        
        # Base goal is offset from object by safe radius
        base_goal_2d = goal_2d - approach_direction * safe_radius
        
        # Adjust parameters based on object type
        switch_distance, velocity = self._get_mission_params(goal)
        
        return MissionConfig(
            base_goal_2d=base_goal_2d,
            arm_goal_3d=arm_goal_3d,
            switch_distance=switch_distance,
            forward_velocity=velocity
        )
    
    def _get_safe_radius(self, goal: dict) -> float:
        """
        Calculate safe distance from object for base positioning.
        
        Args:
            goal: Goal dictionary
        
        Returns:
            Safe radius in meters
        """
        # Object radius + robot radius
        object_radius = goal.get('radius', 0.3)
        return object_radius + self.robot_radius
    
    def _calculate_approach_direction(self, goal_2d: np.ndarray, 
                                     robot_pos: np.ndarray = None) -> np.ndarray:
        """
        Calculate optimal approach direction for the base.
        
        Args:
            goal_2d: Goal position in 2D [x, y]
            robot_pos: Current robot position [x, y]
        
        Returns:
            Normalized direction vector [x, y]
        """
        if robot_pos is not None:
            # Approach from current position direction
            direction = goal_2d - robot_pos
        else:
            # Default: approach from origin
            direction = goal_2d.copy()
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0])  # Default direction
        
        return direction
    
    def _get_mission_params(self, goal: dict) -> tuple:
        """
        Get mission parameters based on object type.
        
        Args:
            goal: Goal dictionary
        
        Returns:
            (switch_distance, forward_velocity) tuple
        """
        object_type = goal.get('object_type', 'default')
        
        params = {
            'furniture': (0.1, 1.2),  # Larger switch distance, slower
            'small_object': (0.1, 1.5),  # Smaller switch distance, faster
            'default': (0.1, 1.5)
        }
        
        return params.get(object_type, params['default'])

