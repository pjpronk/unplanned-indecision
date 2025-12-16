import numpy as np
import math
import random
from typing import List, Dict, Any, Tuple

Point = Tuple[float, float]

class Node:
    def __init__(self, p: Point):
        self.p = p
        self.parent = None


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
        """Plan a path from start to target using RRT (2D).

        Keeps the same signature and return as previous `plan_path`:
        returns a list of 2D waypoints (or None on failure).

        Missing RRT parameters are sourced from the instance with sensible
        defaults if attributes are not present.
        """
        # Parameters (use class attributes if available)
        obstacles = getattr(self, "obstacles", self.obstacles)
        bounds = getattr(self, "bounds", (-3.0, 3.0, -3.0, 3.0))
        step_size = getattr(self, "step_size", self.step_size)
        goal_sample_rate = getattr(self, "goal_sample_rate", 0.10)
        goal_tolerance = getattr(self, "goal_threshold", self.goal_threshold)
        max_iters = getattr(self, "max_iterations", self.max_iterations)
        robot_radius = getattr(self, "robot_radius", 0.0)
        seed = getattr(self, "seed", None)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        start_p = RRTPlanner._as_point(start)
        goal_p = RRTPlanner._as_point(target)
        xmin, xmax, ymin, ymax = bounds

        # Quick validity checks
        if self._segment_hits_any_obstacle(start_p, start_p, obstacles, robot_radius):
            raise ValueError("Start is in collision.")
        if self._segment_hits_any_obstacle(goal_p, goal_p, obstacles, robot_radius):
            raise ValueError("Goal is in collision.")

        nodes: List[Node] = [Node(start_p)]

        for _ in range(max_iters):
            # Sample
            if random.random() < goal_sample_rate:
                q_rand = goal_p
            else:
                q_rand = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))

            # Extend
            i_near = self._nearest(nodes, q_rand)
            q_near = nodes[i_near].p
            q_new = self._steer(q_near, q_rand, step_size)

            # Collision check edge
            if self._segment_hits_any_obstacle(q_near, q_new, obstacles, robot_radius):
                continue

            # append node
            nodes.append(Node(q_new))
            nodes[-1].parent = i_near
            new_i = len(nodes) - 1

            # Try connect to goal
            if RRTPlanner._dist(q_new, goal_p) <= goal_tolerance:
                if not self._segment_hits_any_obstacle(q_new, goal_p, obstacles, robot_radius):
                    nodes.append(Node(goal_p))
                    nodes[-1].parent = new_i
                    goal_i = len(nodes) - 1

                    # Reconstruct waypoints
                    waypoints: List[Point] = []
                    cur = goal_i
                    while cur is not None:
                        waypoints.append(nodes[cur].p)
                        cur = nodes[cur].parent
                    waypoints.reverse()

                    # store and return as list of numpy arrays for compatibility
                    path = [np.array(p) for p in waypoints]
                    self.current_path = path
                    return path

        # failed to find path
        return None

    @staticmethod
    def _as_point(xy) -> Point:
        # Handles tuples/lists/numpy arrays
        return (float(xy[0]), float(xy[1]))


    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))


    @staticmethod
    def _point_in_aabb(p: Point, center: Point, half_w: float, half_h: float) -> bool:
        return (abs(p[0] - center[0]) <= half_w) and (abs(p[1] - center[1]) <= half_h)


    @staticmethod
    def _seg_circle_collision(a: Point, b: Point, c: Point, r: float) -> bool:
        # Segment AB vs circle centered at C radius r
        ax, ay = a
        bx, by = b
        cx, cy = c

        abx, aby = (bx - ax), (by - ay)
        acx, acy = (cx - ax), (cy - ay)
        ab2 = abx * abx + aby * aby

        if ab2 == 0.0:
            return RRTPlanner._dist(a, c) <= r

        t = (acx * abx + acy * aby) / ab2
        t = RRTPlanner._clamp(t, 0.0, 1.0)
        closest = (ax + t * abx, ay + t * aby)
        return RRTPlanner._dist(closest, c) <= r


    def _seg_aabb_collision(self, a: Point, b: Point, center: Point, half_w: float, half_h: float) -> bool:
        # Segment vs axis-aligned bounding box using "slab" intersection
        # Expand: also catches start/end inside box.
        if RRTPlanner._point_in_aabb(a, center, half_w, half_h) or RRTPlanner._point_in_aabb(b, center, half_w, half_h):
            return True

        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay

        # Box min/max
        minx, maxx = center[0] - half_w, center[0] + half_w
        miny, maxy = center[1] - half_h, center[1] + half_h

        tmin, tmax = 0.0, 1.0

        # X slab
        if abs(dx) < 1e-12:
            if ax < minx or ax > maxx:
                return False
        else:
            tx1 = (minx - ax) / dx
            tx2 = (maxx - ax) / dx
            t1, t2 = (tx1, tx2) if tx1 <= tx2 else (tx2, tx1)
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False

        # Y slab
        if abs(dy) < 1e-12:
            if ay < miny or ay > maxy:
                return False
        else:
            ty1 = (miny - ay) / dy
            ty2 = (maxy - ay) / dy
            t1, t2 = (ty1, ty2) if ty1 <= ty2 else (ty2, ty1)
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False

        # If we got here, segment intersects box
        return True


    def _segment_hits_any_obstacle(self, a: Point, b: Point, obstacles: List[Dict[str, Any]], robot_radius: float = 0.0) -> bool:
        for obs in obstacles:
            t = obs.get("type", "").lower()
            pos = RRTPlanner._as_point(obs["position"])

            if t in ("sphere", "cylinder", "circle"):
                r = float(obs["radius"]) + robot_radius
                if RRTPlanner._seg_circle_collision(a, b, pos, r):
                    return True

            elif t == "box":
                # axis-aligned box centered at position
                # width (x), length (y)
                half_w = 0.5 * float(obs["width"]) + robot_radius
                half_h = 0.5 * float(obs["length"]) + robot_radius
                if self._seg_aabb_collision(a, b, pos, half_w, half_h):
                    return True

            else:
                raise ValueError(f"Unknown obstacle type: {t}")

        return False


    def _steer(self, from_p: Point, to_p: Point, step_size: float) -> Point:
        d = RRTPlanner._dist(from_p, to_p)
        if d <= step_size:
            return to_p
        ux, uy = (to_p[0] - from_p[0]) / d, (to_p[1] - from_p[1]) / d
        return (from_p[0] + step_size * ux, from_p[1] + step_size * uy)


    def _nearest(self, nodes: List[Node], p: Point) -> int:
        best_i = 0
        best_d = float("inf")
        for i, n in enumerate(nodes):
            d = RRTPlanner._dist(n.p, p)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i