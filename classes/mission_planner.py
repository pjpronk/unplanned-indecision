import numpy as np
from dataclasses import dataclass
import math


@dataclass
class MissionConfig:
    """Mission parameters for navigation and reaching."""

    base_goal_2d: tuple
    arm_goal_3d: tuple


class MissionPlanner:
    """
    Plans missions from environment goals.
    Computes safe base positions and generates MissionConfig objects.
    """

    def __init__(self, robot_radius, obstacles_2d):
        self.robot_radius = robot_radius
        self.obstacles_2d = obstacles_2d or []

    def plan_missions(self, goals):
        """
        Plans missions for multiple goals.

        Args:
            goals: List of 3-tuples (x, y, z) representing goal positions
        Returns:
            List of MissionConfig objects, one per successfully planned goal
        """
        missions = []
        for goal in goals:
            mission = self._plan_single_mission(goal)
            if mission:
                missions.append(mission)
        return missions

    def _plan_single_mission(self, goal):
        """
        Plans a single mission by computing base position and creating MissionConfig
        """
        arm_goal_3d = goal
        base_goal_2d = tuple(self._better_landing_zones(goal)[0])

        return MissionConfig(
            base_goal_2d=base_goal_2d,
            arm_goal_3d=arm_goal_3d,
        )

    def _better_landing_zones(self, goal):
        """
        Generate and rank base goal positions around the object.
        Returns a list sorted from best to worst.
        """

        goal_2d = np.array(goal[:2])
        safe_radius = self.robot_radius + 0.5

        # Front-biased sampling (relative to world x-axis)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 15)

        candidates = []

        for angle in angles:
            direction = np.array([math.cos(angle), math.sin(angle)])
            base_goal = goal_2d - direction * safe_radius

            if not (-0.5 < base_goal[0] < 5 and -0.5 < base_goal[1] < 6):
                continue

            if self._check_collision(base_goal):
                continue

            clearance = self._forward_clearance(base_goal, goal_2d, 2.0)
            score = clearance

            candidates.append((score, base_goal))

        if not candidates:
            return []

        # Sort best first
        candidates.sort(key=lambda x: x[0], reverse=True)

        return [c[1] for c in candidates]

    def _forward_clearance(self, base, goal_2d, max_dist):
        """
        Measures free space between base and object along approach direction.
        """

        direction = goal_2d - base
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            return 0.0

        direction /= dist
        step = 0.05
        traveled = 0.0

        while traveled < min(dist, max_dist):
            p = base + traveled * direction
            if self._check_collision(p):
                break
            traveled += step

        return traveled

    #
    def _check_collision(self, base_goal):
        """
        Checks if base position collides with any obstacle (sphere, cylinder, or box)
        """

        x = float(base_goal[0])
        y = float(base_goal[1])
        r_robot = float(self.robot_radius)

        for obs in self.obstacles_2d:
            ox, oy = obs["position"][:2]

            if obs.get("type") in ["sphere", "cylinder", "circle"]:
                r = obs["radius"] + r_robot
                if (x - ox) ** 2 + (y - oy) ** 2 <= r**2:
                    return True

            elif obs.get("type") == "box":
                hw = 0.5 * obs["width"] + r_robot
                hl = 0.5 * obs["length"] + r_robot
                if abs(x - ox) <= hw and abs(y - oy) <= hl:
                    return True

        return False
