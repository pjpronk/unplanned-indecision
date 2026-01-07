import numpy as np

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle


class PlaygroundEnv:
    """
    Environment that manages obstacles and provides both 3D PyBullet simulation 
    and 2D obstacle data for RRT path planning.
    
    Can generate either predefined or random obstacles with collision avoidance.
    """

    def __init__(self, env, end_pos, robot_radius, random=False, obstacle_count=25):
        """
        Initialize PlaygroundEnv obstacle manager.
        
        Args:
            env: UrdfEnv instance to populate with obstacles
            end_pos: End position as tuple (x, y, z)
            robot_radius: Robot radius for collision checking
            random: Generate random obstacles if True, predefined otherwise
            obstacle_count: Number of obstacles to generate (random mode only)
        """
        self.env = env
        self.random = random
        self.obstacle_count = obstacle_count
        self.obstacles_3d = []
        self.end_pos = end_pos
        self.r_robot = robot_radius
        
        self.setup_obstacles()

    def setup_obstacles(self):
        """
        Set up obstacles in the environment.
        
        If random=True: generates random boxes and cylinders with collision avoidance
        If random=False: creates predefined scenario with box, lego pile, and duplo wall
        """
        if self.random:
            self._setup_random_obstacles()
                
        else:
            self._setup_predefined_obstacles()

    def _setup_random_obstacles(self):
        """Generate random boxes and cylinders with collision avoidance."""
        obstacles_counter = 0
        
        # Try to place boxes
        for i in range(self.obstacle_count):
            if self._try_place_random_box(f"box_{i}"):
                obstacles_counter += 1
        
        # Fill remaining quota with cylinders
        remaining = self.obstacle_count - obstacles_counter
        for i in range(remaining):
            self._try_place_random_cylinder(f"cylinder_{i + obstacles_counter}")

    def _try_place_random_box(self, name: str, max_attempts: int = 50) -> bool:
        """Try to place a random box obstacle without collisions. Returns True if successful."""
        box_template = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [4.5, 2.25, 0.15],
                "width": 1.2,
                "height": 0.3,
                "length": 2.2,
            },
            "low": {
                "position": [2, 2, 0],
                "width": 0.1,
                "height": 0.1,
                "length": 0.1,
            },
            "high": {
                "position": [9, 5, 0.3],
                "width": 2.0,
                "height": 1.0,
                "length": 2.0,
            },
            "rgba": [0.4, 0.4, 0.2, 1.0]
        }
        
        for _ in range(max_attempts):
            box_obst = BoxObstacle(name=name, content_dict=box_template)
            box_obst.shuffle()
            
            if not self._has_collision_with_existing(box_obst):
                self.obstacles_3d.append(box_obst)
                self.env.add_obstacle(box_obst)
                return True
        
        return False

    def _try_place_random_cylinder(self, name: str, max_attempts: int = 100) -> bool:
        """Try to place a random cylinder obstacle without collisions. Returns True if successful."""
        cylinder_template = {
            "type": "cylinder",
            "movable": False,
            "geometry": {
                "position": [4.5, 2.25, 0.15],
                "radius": 1.2,
                "height": 0.3,
            },
            "low": {
                "position": [2, 2, 0],
                "radius": 0.02,
                "height": 0.1,
            },
            "high": {
                "position": [9, 5, 0.3],
                "radius": 0.2,
                "height": 3.0,
            },
            "rgba": [0.8, 0.4, 0.2, 1.0]
        }
        
        for _ in range(max_attempts):
            cyl_obst = CylinderObstacle(name=name, content_dict=cylinder_template)
            cyl_obst.shuffle()
            
            if not self._has_collision_with_existing(cyl_obst):
                self.obstacles_3d.append(cyl_obst)
                self.env.add_obstacle(cyl_obst)
                return True
        
        return False

    def _has_collision_with_existing(self, new_obstacle) -> bool:
        """Check if new obstacle collides with any existing obstacles."""
        for existing in self.obstacles_3d:
            if self.collision_check(new_obstacle, existing):
                return True
        return False

    def _setup_predefined_obstacles(self):
        """Create predefined obstacle scenario with box, lego pile, and duplo wall."""
        # Large rotated box obstacle
        box_obst_dict_1 = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [3, 2, 0.15],
                "orientation": [0.5, 0, 0, 0.707],
                "width": 0.8,
                "height": 1,
                "length": 1.8,
            },
            "rgba": [0.8, 0.2, 0.2, 1.0]
        }
        box_obst = BoxObstacle(name="box_1", content_dict=box_obst_dict_1)
        self.obstacles_3d.append(box_obst)
        self.env.add_obstacle(box_obst)

        # Lego pile (7 small movable boxes)
        self._create_lego_pile()
        
        # Duplo wall (pyramid structure)
        self._create_duplo_wall()

    def _create_lego_pile(self):
        """Create scattered lego pieces near target area."""
        random_offsets = np.random.uniform(-0.5, 0.5, 100)
        
        for i in range(7):
            # Try up to 50 times to place each lego without collision
            for attempt in range(50):
                lego = {
                    "type": "box",
                    "movable": True,
                    "geometry": {
                        "position": [
                            float(4 + np.random.choice(random_offsets)),
                            float(1 + np.random.choice(random_offsets)),
                            0.05 + i * 0.02
                        ],
                        "orientation": [float(np.random.choice(random_offsets)), 0, 0, 1],
                        "width": 0.2,
                        "height": 0.1,
                        "length": 0.08,
                    },
                    "rgba": [0.4, 0.2, 0.2, 1.0]
                }
                
                box_obst = BoxObstacle(name=f"lego_{i}", content_dict=lego)
                
                if not self._has_collision_with_existing(box_obst):
                    self.obstacles_3d.append(box_obst)
                    self.env.add_obstacle(box_obst)
                    break

    def _create_duplo_wall(self):
        """Create pyramid-shaped duplo wall structure."""
        for j in range(3):
            for i in range(3 - j):
                duplo = {
                    "type": "box",
                    "movable": False,
                    "geometry": {
                        "position": [
                            6.0,
                            1 + i * 0.2 + j * 0.1,
                            0.1 + j * 0.2
                        ],
                        "orientation": [0, 0, 0, 1],
                        "width": 0.4,
                        "height": 0.2,
                        "length": 0.2,
                    },
                    "rgba": [0.2, 0.2, 0.8, 1.0]
                }
                box_obst = BoxObstacle(name=f"duplo_{i}-{j}", content_dict=duplo)
                self.obstacles_3d.append(box_obst)
                self.env.add_obstacle(box_obst)

    def get_2d_obstacles(self):
        """
        Get 2D projection of obstacles for path planning.
        Projects 3D obstacles onto the x-y plane.
        
        Returns:
            list: List of dictionaries containing 2D obstacle information
                  Each dict contains: name, type, position (x,y), and size info
        """
        obstacles_2d = []
        
        for obstacle in self.obstacles_3d:
            # Use getattr with fallback to avoid AttributeError
            obs_2d = {
                'name': getattr(obstacle, 'name', getattr(obstacle, '_name', 'obstacle')),
                'type': obstacle.type(),
            }
            
            # Get 3D position and project to 2D (x, y plane, ignoring z)
            position_3d = obstacle.position()
            obs_2d['position'] = np.array([position_3d[0], position_3d[1]])
            
            # Extract size parameters based on type
            if obstacle.type() == 'sphere':
                obs_2d['radius'] = obstacle.radius()
                
            elif obstacle.type() == 'box':
                obs_2d['width'] = obstacle.width()
                obs_2d['length'] = obstacle.length()
                obs_2d['bounding_radius'] = self.bounding_circle_radius(obstacle)
                
            elif obstacle.type() == 'cylinder':
                obs_2d['radius'] = obstacle.radius()
            
            obstacles_2d.append(obs_2d)
        
        return obstacles_2d
    
    def bounding_circle_radius(self, object):
        """
        Calculate conservative bounding circle radius for an obstacle.
        Used for quick collision filtering.
        """
        if object.type() == 'box':
            return np.sqrt(object.width()**2 + object.length()**2) / 2
        elif object.type() == 'cylinder':
            return object.radius()
        else:
            raise ValueError(f"Unknown obstacle type: {object.type()}")
        
    def better_bound(self, object):
        """
        Get bounding points for an obstacle, expanded by robot radius.
        
        For boxes: returns 4 corner points
        For cylinders: returns 10 points around the circumference
        """
        center = np.array(object.position())
        
        if object.type() == "box":
            hw = object.width() / 2 + self.r_robot   # half-width + robot radius
            hl = object.length() / 2 + self.r_robot  # half-length + robot radius
            corners = [
                center + np.array([hw, hl, 0]),    # top right
                center + np.array([hw, -hl, 0]),   # bottom right
                center + np.array([-hw, -hl, 0]),  # bottom left
                center + np.array([-hw, hl, 0])    # top left
            ]
            return corners
        
        elif object.type() == "cylinder":
            points = []
            r = object.radius() + self.r_robot
            for angle in np.linspace(0, 2*np.pi, num=10, endpoint=False):
                point = center + np.array([
                    r * np.cos(angle),
                    r * np.sin(angle),
                    0
                ])
                points.append(point)
            return points
        
        else:
            raise ValueError(f"Unknown obstacle type: {object.type()}")
        
    def collision_check(self, object1, object2) -> bool:
        """
        Check if two objects collide using AABB (Axis-Aligned Bounding Box) test.
        Returns True if objects overlap, False otherwise.
        """
        points_1 = np.array(self.better_bound(object1))
        points_2 = np.array(self.better_bound(object2))

        # Vectorized AABB calculation (much faster than list comprehensions)
        x_min_1, y_min_1 = points_1[:, :2].min(axis=0)
        x_max_1, y_max_1 = points_1[:, :2].max(axis=0)
        
        x_min_2, y_min_2 = points_2[:, :2].min(axis=0)
        x_max_2, y_max_2 = points_2[:, :2].max(axis=0)

        # No overlap if separated on ANY axis
        if x_min_1 > x_max_2 or x_max_1 < x_min_2 or y_min_1 > y_max_2 or y_max_1 < y_min_2:
            return False
        
        return True
