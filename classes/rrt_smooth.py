import numpy as np
import math
import random
from typing import List, Dict, Any, Tuple

Point = Tuple[float, float]

class Node:
    def __init__(self, p: Point, parent=None, cost: float = 0.0):
        self.p = p
        self.parent = parent
        self.cost = cost  # cost-to-come from start



class RRTPlanner:
    """
    Pure RRT path planner - only does planning, no action computation.
    Independent of path following.
    """

    def __init__(self, obstacles, step_size=0.1, max_iterations=1000, 
                 goal_threshold=0.1, bounds=(-3.0, 3.0, -3.0, 3.0), 
                 robot_radius=0.3, goal_sample_rate=0.10):
        """
        Initialize RRT planner.
        
        Args:
            obstacles: List of 2D obstacle dictionaries from ObstacleManager.get_2d_obstacles()
            step_size: Step size for RRT tree extension
            max_iterations: Maximum iterations for RRT algorithm
            goal_threshold: Distance threshold to consider goal reached
            bounds: Tuple (x_min, x_max, y_min, y_max) defining planning bounds
            robot_radius: Radius of the robot for collision checking
            goal_sample_rate: Probability of sampling goal directly (0.0-1.0)
        """
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.bounds = bounds
        self.robot_radius = robot_radius
        self.goal_sample_rate = goal_sample_rate
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
        robot_radius = getattr(self, "robot_radius", 0.25)
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

        nodes: List[Node] = [Node(start_p, parent=None, cost=0.0)]


        best_path = None
        best_cost = None

        for _ in range(max_iters):
            # Sample
            if random.random() < goal_sample_rate:
                q_rand = goal_p
            else:
                if best_cost is None:
                    # No solution yet: sample globally
                    q_rand = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
                else:
                    # Have a solution: sample inside the informed ellipse
                    q_rand = self._sample_in_informed_ellipse(start_p, goal_p, best_cost)

            # Extend
            i_near = self._nearest(nodes, q_rand)
            q_near = nodes[i_near].p
            q_new = self._steer(q_near, q_rand, step_size)

            # Collision check edge
            if self._segment_hits_any_obstacle(q_near, q_new, obstacles, robot_radius):
                continue

            # append node
                        # --- RRT* radius (2D) ---
            n = len(nodes)
            gamma = getattr(self, "rrt_star_gamma", 1.0)  # tuning knob
            radius = gamma * math.sqrt(max(math.log(n + 1) / (n + 1), 1e-9))
            radius = min(radius, 5.0 * step_size)  # practical cap

            near_inds = self._near(nodes, q_new, radius)
            if i_near not in near_inds:
                near_inds.append(i_near)

            # Choose best parent among near nodes (collision-free)
            best_parent = i_near
            best_cost_to_new = nodes[i_near].cost + RRTPlanner._dist(nodes[i_near].p, q_new)

            for j in near_inds:
                cand_cost = nodes[j].cost + RRTPlanner._dist(nodes[j].p, q_new)
                if cand_cost < best_cost_to_new:
                    if not self._segment_hits_any_obstacle(nodes[j].p, q_new, obstacles, robot_radius):
                        best_parent = j
                        best_cost_to_new = cand_cost

            nodes.append(Node(q_new, parent=best_parent, cost=best_cost_to_new))
            new_i = len(nodes) - 1


            # Rewire neighbors through the new node if cheaper
            for j in near_inds:
                if j == new_i:
                    continue
                new_cost_to_j = nodes[new_i].cost + RRTPlanner._dist(nodes[new_i].p, nodes[j].p)
                if new_cost_to_j < nodes[j].cost:
                    if not self._segment_hits_any_obstacle(nodes[new_i].p, nodes[j].p, obstacles, robot_radius):
                        nodes[j].parent = new_i
                        nodes[j].cost = new_cost_to_j


            # Try connect to goal
# Try connect to goal (update best solution, but keep searching)
            if RRTPlanner._dist(q_new, goal_p) <= goal_tolerance:
                if not self._segment_hits_any_obstacle(q_new, goal_p, obstacles, robot_radius):
                    # Temporarily add goal to reconstruct a candidate path
                                        # Reconstruct candidate waypoints by backtracking from new_i, then add goal
                    waypoints: List[Point] = []
                    cur = new_i
                    seen = set()
                    while cur is not None:
                        if cur in seen:
                            waypoints = None
                            break
                        seen.add(cur)
                        waypoints.append(nodes[cur].p)
                        cur = nodes[cur].parent

                    if waypoints is None:
                        continue
                    waypoints.reverse()
                    waypoints.append(goal_p)

                    candidate_cost = RRTPlanner._path_cost(waypoints)

                    if (best_cost is None) or (candidate_cost < best_cost):
                        best_cost = candidate_cost
                        best_path = [np.array(p) for p in waypoints]
                        self.current_path = best_path

                    # NOTE: keep the goal node in the tree (simple + works fine)
                    # and continue searching for improvements
                    continue

        # failed to find path
                # Smooth best path (if any) before returning
        if best_path is not None:
            raw_len = RRTPlanner._path_length(best_path)
            best_path = self._shortcut_smooth(best_path, obstacles, robot_radius)
            smooth_len = RRTPlanner._path_length(best_path)

            print(f"[RRT] Raw path length:     {raw_len:.3f} m")
            print(f"[RRT] Smoothed length:     {smooth_len:.3f} m")

        return best_path



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

    def _near(self, nodes: List[Node], p: Point, radius: float) -> List[int]:
        r2 = radius * radius
        out = []
        for i, n in enumerate(nodes):
            dx = n.p[0] - p[0]
            dy = n.p[1] - p[1]
            if dx * dx + dy * dy <= r2:
                out.append(i)
        return out


    def _nearest(self, nodes: List[Node], p: Point) -> int:
        best_i = 0
        best_d = float("inf")
        for i, n in enumerate(nodes):
            d = RRTPlanner._dist(n.p, p)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    @staticmethod
    def _path_cost(waypoints: List[Point]) -> float:
            """Sum of Euclidean segment lengths along a waypoint list."""
            if len(waypoints) < 2:
                return 0.0
            c = 0.0
            for i in range(len(waypoints) - 1):
                c += RRTPlanner._dist(waypoints[i], waypoints[i + 1])
            return c

    def _sample_in_informed_ellipse(self, start: Point, goal: Point, c_best: float) -> Point:
            """
            Sample uniformly-ish inside the 2D informed set (ellipse):
                dist(start, x) + dist(x, goal) <= c_best

            This uses a standard trick:
            - sample uniformly in unit disk
            - scale by ellipse radii
            - rotate to align with start->goal
            - translate to ellipse center
            """
            # Minimum possible path cost is straight-line distance
            c_min = RRTPlanner._dist(start, goal)

            # If we don't have slack (or numerical issues), fall back to goal-biased-ish sampling
            if c_best <= c_min + 1e-12:
                return goal

            # Ellipse parameters
            center = ((start[0] + goal[0]) * 0.5, (start[1] + goal[1]) * 0.5)

            # Semi-major axis a and semi-minor axis b
            a = c_best * 0.5
            b = math.sqrt(max(a * a - (c_min * 0.5) ** 2, 0.0))

            # Rotation to align ellipse with the line start->goal
            theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            # Sample uniformly in unit disk (r = sqrt(u) trick)
            u = random.random()
            v = random.random()
            r = math.sqrt(u)
            ang = 2.0 * math.pi * v
            xd = r * math.cos(ang)
            yd = r * math.sin(ang)

            # Scale to ellipse
            xe = a * xd
            ye = b * yd

            # Rotate + translate
            x = cos_t * xe - sin_t * ye + center[0]
            y = sin_t * xe + cos_t * ye + center[1]
            return (x, y)
    
    def _shortcut_smooth(
        self,
        path: List[np.ndarray],
        obstacles: List[Dict[str, Any]],
        robot_radius: float,
        iters: int = 200
    ) -> List[np.ndarray]:
        """
        Simple shortcut smoother:
        repeatedly tries to replace path segments with a straight line if collision-free.
        """
        if path is None or len(path) < 3:
            return path

        # Convert to tuple points for collision checks
        pts: List[Point] = [RRTPlanner._as_point(p) for p in path]

        for _ in range(iters):
            if len(pts) < 3:
                break

            # pick two indices with at least one point between them
            i = random.randint(0, len(pts) - 3)
            j = random.randint(i + 2, len(pts) - 1)

            # If we can connect pts[i] directly to pts[j], remove the middle points
            if not self._segment_hits_any_obstacle(pts[i], pts[j], obstacles, robot_radius):
                pts = pts[: i + 1] + pts[j:]

        return [np.array(p) for p in pts]


    def _path_length(path):
        """Total Euclidean length of a 2D path."""
        if path is None or len(path) < 2:
            return 0.0
        import math
        dist = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            dist += math.hypot(dx, dy)
        return dist

            