import numpy as np


class PathFollower:
    """
    Path follower that computes actions to follow a given path.
    """

    def __init__(self, forward_velocity, waypoint_threshold):
        """
        Initialize path follower.

        Args:
            forward_velocity: Base forward velocity (m/s)
            waypoint_threshold: Distance threshold to consider waypoint reached
        """
        self.forward_velocity = forward_velocity
        self.waypoint_threshold = waypoint_threshold
        self.current_waypoint_idx = 0

    def follow(self, current_pos, path, action_size, final_target=None):
        """
        Compute action to follow the path.

        Args:
            current_pos: Current 2D position [x, y]
            path: List of 2D waypoints to follow
            action_size: Size of action vector to return
            final_target: Optional 3D target position [x, y, z] for final orientation

        Returns:
            np.array: Action vector for the robot
        """
        # Handle empty or invalid path
        if path is None or len(path) == 0:
            return np.zeros(action_size)

        # Ensure current_pos is a numpy array
        current_pos = np.array(current_pos)

        # Get current target waypoint
        target_waypoint = self._get_current_waypoint(path)

        # Check if we should advance to next waypoint
        if self._should_advance_waypoint(current_pos, target_waypoint):
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(path):
                # Reached end of path
                return np.zeros(action_size)
            target_waypoint = self._get_current_waypoint(path)

        # Check if we're at the last waypoint
        is_last_waypoint = self.current_waypoint_idx >= len(path) - 1

        # Compute action to move toward waypoint
        action = self._compute_action_to_waypoint(
            current_pos, target_waypoint, action_size, final_target, is_last_waypoint
        )

        return action

    def reset(self):
        """Reset the path follower state."""
        self.current_waypoint_idx = 0

    def _get_current_waypoint(self, path):
        """Get the current waypoint to navigate to."""
        if self.current_waypoint_idx >= len(path):
            return path[-1]  # Return last waypoint if we're at the end
        return path[self.current_waypoint_idx]

    def _should_advance_waypoint(self, current_pos, target_waypoint):
        """Check if we should advance to next waypoint."""
        distance = np.linalg.norm(current_pos - target_waypoint)
        return distance < self.waypoint_threshold

    def _compute_action_to_waypoint(
        self, current_pos, target_waypoint, action_size, final_target=None, is_last_waypoint=False
    ):
        """
        Compute action to move toward target waypoint.
        Basic proportional controller for mobile base.

        Args:
            current_pos: Current 2D position [x, y]
            target_waypoint: Target 2D waypoint [x, y]
            action_size: Size of action vector
            final_target: Optional 3D target position [x, y, z] for final orientation
            is_last_waypoint: Whether we're approaching the last waypoint

        Returns:
            np.array: Action vector
        """
        action = np.zeros(action_size)

        # Compute direction vector to waypoint
        direction = target_waypoint - current_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-3:
            return action

        # Normalize direction
        direction = direction / distance

        action[0] = direction[0] * self.forward_velocity
        action[1] = direction[1] * self.forward_velocity

        # Use final target for orientation if we're at the last waypoint
        if final_target is not None and is_last_waypoint:
            # Orient towards the final 3D target (arm goal)
            final_direction = np.array(final_target[:2]) - current_pos
            angle_to_target = np.arctan2(final_direction[1], final_direction[0])
        else:
            # Orient towards the next waypoint
            angle_to_target = np.arctan2(direction[1], direction[0])

        action[2] = angle_to_target * 0.5

        return action
