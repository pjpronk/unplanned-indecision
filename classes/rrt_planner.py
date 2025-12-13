import numpy as np


class RRTPlanner:
    """
    Pure RRT path planner - only does planning, no action computation.
    Independent of path following.
    """

    def __init__(self, obstacles, step_size=0.1, max_iterations=1000, 
                 goal_threshold=0.1):
        """
        Initialize RRT planner.
        
        Args:
            obstacles: List of 2D obstacle dictionaries from ObstacleManager.get_2d_obstacles()
            step_size: Step size for RRT tree extension
            max_iterations: Maximum iterations for RRT algorithm
            goal_threshold: Distance threshold to consider goal reached
        """
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.current_path = None    
    
    def plan_path(self, start, target):
        """
        Plan a path from start to target using a straight line.
        
        Args:
            start: 2D start position [x, y]
            target: 2D target position [x, y]
            
        Returns:
            list: Path as list of 2D waypoints, or None if no path found
        """
        start_2d = np.array(start[:2]) if len(start) > 2 else np.array(start)
        goal_2d = np.array(target[:2]) if len(target) > 2 else np.array(target)
        
        # Create waypoints along the straight line
        distance = np.linalg.norm(goal_2d - start_2d)
        num_waypoints = int(distance / self.step_size) + 1
        
        path = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints if num_waypoints > 0 else 1.0
            waypoint = start_2d + t * (goal_2d - start_2d)
            path.append(waypoint)
        
        self.current_path = path
        return path

