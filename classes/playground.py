import numpy as np

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle


class PlaygroundEnv:
    """
    Environment that manages obstacles and provides both 3D PyBullet simulation
    and 2D obstacle data for RRT path planning.

    Creates predefined obstacle scenarios.
    """

    def __init__(self, env, end_pos, robot_radius, type):
        """
        Initialize PlaygroundEnv obstacle manager.

        Args:
            env: UrdfEnv instance to populate with obstacles
            end_pos: End position as tuple (x, y, z)
            robot_radius: Robot radius for collision checking
            type: Predefined scenario type
        """
        self.env = env
        self.type = type
        self.obstacles_3d = []
        self.end_pos = end_pos
        self.r_robot = robot_radius
        self.chair_marker = []
        self.closet_marker = []

        self.setup_obstacles()

    def setup_obstacles(self):
        """Set up obstacles in the environment using predefined scenarios."""
        self._setup_predefined_obstacles(type=self.type)

    # Pre defined enviroment

    def _setup_predefined_obstacles(self, type="medium"):
        """Create predefined obstacle scenario with box, lego pile, and duplo wall."""

        if type == "medium":
            # Lego pile (7 small movable boxes)
            self._create_lego_pile()

            # Duplo wall (pyramid structure)
            self._create_duplo_wall([1.5, 0.5, 0])

            # # Chair obstacle
            # pos_chair = [2,4]
            # self._create_chair_obstacle(pos_chair, name_prefix="chair1")
            # self.chair_marker = [pos_chair[0], pos_chair[1], 0.1]
            # Closet obstacle
            pos_closet = [4, 0, 0.05]
            self._create_closet_obstacle(pos_closet)
            self.closet_marker = [pos_closet[0], pos_closet[1], 0.6]

            # second goal
            self.chair_marker = [4, 3, 0.1]

            # Walls
            self._create_walls()

            # Football obstacle
            self._create_football_obstacle([3.0, 3.0, 0.11])

            # toycar obstacle
            self._create_toycar_obstacle([4.0, 3.5, 0.05])

        if type == "hard":
            # Lego pile (7 small movable boxes)
            self._create_lego_pile()

            # Duplo wall (pyramid structure)
            self._create_duplo_wall([1.5, 0.5, 0])
            self._create_duplo_wall([1, 3, 0])

            # Chair obstacle
            pos_chair = [2, 4]
            self._create_chair_obstacle(pos_chair, name_prefix="chair1")
            self.chair_marker = [pos_chair[0], pos_chair[1], 0.1]

            # Closet obstacle
            pos_closet = [4, 0, 0.05]
            self._create_closet_obstacle(pos_closet)
            self.closet_marker = [pos_closet[0], pos_closet[1], 0.6]

            # Walls
            self._create_walls()

            # Football obstacle
            self._create_football_obstacle([3.0, 3.0, 0.11])
            self._create_football_obstacle([0, 1.5, 0.11])

            # toycar obstacle
            self._create_toycar_obstacle([4.0, 3.5, 0.05])
            self._create_toycar_obstacle([5.0, 1, 0.05])

        if type == "easy":
            # Lego pile (7 small movable boxes)
            self._create_lego_pile()

            # Duplo wall (pyramid structure)
            self._create_duplo_wall([1.5, 0.5, 0])

            # Football obstacle
            self._create_football_obstacle([3.0, 3.0, 0.11])

            # toycar obstacle
            self._create_toycar_obstacle([4.0, 3.5, 0.05])
            self._create_toycar_obstacle([5.0, 2, 0.05])
            self._create_toycar_obstacle([2.0, 4, 0.05])

            # wall
            self._create_walls()

            # targets
            target_1 = [2, 1, 0.0]

            target_2 = [4, 2, 0.0]
            self.chair_marker = target_1
            self.closet_marker = target_2

        if type == "test":
            target_1 = [1, 1, 0.5]
            self.chair_marker = target_1

            self._create_football_obstacle([1.5, 1.5, 0.11])

            target_2 = [2, 2, 0.0]
            self.closet_marker = target_2

        if type == "quantum_RRT":
            self._create_duplo_wall([1.5, -0.7, 0])
            self._create_duplo_wall([1.5, 0.5, 0])
            self._create_duplo_wall([1.5, 2, 0])
            self._create_duplo_wall([1.5, 3.8, 0])
            self._create_duplo_wall([1.5, 5, 0])
            self._create_walls()

            target_1 = [4, 1, 0.0]
            self.chair_marker = target_1
            target_2 = [4, 4, 0.0]
            self.closet_marker = target_2

    # def _create_lego_pile(self):
    #     """Create scattered lego pieces near target area."""
    #     random_offsets = np.random.uniform(-0.5, 0.5, 100)

    #     for i in range(7):
    #         # Try up to 50 times to place each lego without collision
    #         for attempt in range(50):
    #             lego = {
    #                 "type": "box",
    #                 "movable": True,
    #                 "geometry": {
    #                     "position": [
    #                         float(3 + np.random.choice(random_offsets)),
    #                         float(1.5 + np.random.choice(random_offsets)),
    #                         0.05 + i * 0.2
    #                     ],
    #                     "orientation": [float(np.random.choice(random_offsets) * 2), 0, 0, 0.707],
    #                     "width": 0.1,
    #                     "height": 0.05,
    #                     "length": 0.04,
    #                 },
    #                 "rgba": [0.4, 0.2, 0.2, 1.0]
    #             }

    #             box_obst = BoxObstacle(name=f"lego_{i}", content_dict=lego)

    #             if not self._has_collision_with_existing(box_obst):
    #                 self.obstacles_3d.append(box_obst)
    #                 self.env.add_obstacle(box_obst)
    #                 break

    def _create_lego_pile(self):
        """Create scattered lego pieces near target area."""
        random_offsets = np.random.uniform(-0.5, 0.5, 100)

        for i in range(7):
            # Try up to 50 times to place each lego without collision
            lego = {
                "type": "box",
                "movable": True,
                "geometry": {
                    "position": [
                        float(3 + np.random.choice(random_offsets)),
                        float(1.5 + np.random.choice(random_offsets)),
                        0.05 + i * 0.2,
                    ],
                    "orientation": [float(np.random.choice(random_offsets) * 2), 0, 0, 0.707],
                    "width": 0.1,
                    "height": 0.05,
                    "length": 0.04,
                },
                "rgba": [0.4, 0.2, 0.2, 1.0],
            }

            box_obst = BoxObstacle(name=f"lego_{i}", content_dict=lego)
            self.obstacles_3d.append(box_obst)
            self.env.add_obstacle(box_obst)

    def _create_duplo_wall(self, base_pos=(6.0, 1.0, 0.0)):
        """Create pyramid-shaped duplo wall structure."""

        x0, y0, z0 = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])

        for j in range(3):
            for i in range(3 - j):
                duplo = {
                    "type": "box",
                    "movable": False,
                    "geometry": {
                        "position": [
                            float(x0),
                            float(y0 - 0.4 + i * 0.4 + j * 0.2),
                            float(z0 + 0.1 + j * 0.2),
                        ],
                        # "orientation": [0, 0, 0, 1],
                        "width": 0.4,
                        "height": 0.2,
                        "length": 0.2,
                    },
                    "rgba": [0.2, 0.2, 0.8, 1.0],
                }

                box_obst = BoxObstacle(name=f"{'duplo'}_{i}-{j}", content_dict=duplo)
                self.obstacles_3d.append(box_obst)
                self.env.add_obstacle(box_obst)

    def _create_chair_obstacle(self, base_xy=(5.0, 2.5), name_prefix="chair"):
        "god help me , this is gonna give me a headache"

        x0, y0 = float(base_xy[0]), float(base_xy[1])
        h = 0.2
        # Seat
        seat = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [x0, y0, 0.45 + h],
                "width": 0.7,
                "height": 0.1,
                "length": 0.7,
            },
            "rgba": [0.6, 0.4, 0.2, 1.0],
        }
        seat_obs = BoxObstacle(name=f"{name_prefix}_seat", content_dict=seat)
        self.obstacles_3d.append(seat_obs)
        self.env.add_obstacle(seat_obs)

        # Legs
        leg_offsets = [(-0.3, -0.3), (-0.3, 0.3), (0.3, -0.3), (0.3, 0.3)]
        for i, (dx, dy) in enumerate(leg_offsets):
            leg = {
                "type": "cylinder",
                "movable": False,
                "geometry": {
                    "position": [x0 + dx, y0 + dy, 0.225 + h / 2],
                    "radius": 0.03,
                    "height": 0.45 + h,
                },
                "rgba": [0.6, 0.4, 0.2, 1.0],
            }
            leg_obs = CylinderObstacle(name=f"{name_prefix}_leg_{i}", content_dict=leg)
            self.obstacles_3d.append(leg_obs)
            self.env.add_obstacle(leg_obs)

        # back
        back = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [x0 - 0.25, y0, 0.75 + h],
                "width": 0.7,
                "height": 0.6,
                "length": 0.1,
            },
            "rgba": [0.6, 0.4, 0.2, 1.0],
        }
        back_obs = BoxObstacle(name=f"{name_prefix}_back", content_dict=back)
        self.obstacles_3d.append(back_obs)
        self.env.add_obstacle(back_obs)

    def _create_closet_obstacle(self, postion):
        "creates a ikea like simple closet"

        # outer box
        outer_bottom = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": postion,
                "width": 0.6,  # y direction
                "height": 0.1,  # z direction
                "length": 1.2,  # x direction
            },
            "rgba": [0.9, 0.9, 0.9, 1.0],
        }
        outer_obs = BoxObstacle(name="closet_outer", content_dict=outer_bottom)
        self.obstacles_3d.append(outer_obs)
        self.env.add_obstacle(outer_obs)

        outer_right = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [postion[0] + 0.6, postion[1], postion[2] + 0.55],
                "width": 0.6,
                "height": 1.2,
                "length": 0.1,
            },
            "rgba": [0.9, 0.9, 0.9, 1.0],
        }
        outer_right_obs = BoxObstacle(name="closet_outer_right", content_dict=outer_right)
        self.obstacles_3d.append(outer_right_obs)
        self.env.add_obstacle(outer_right_obs)

        outer_left = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [postion[0] - 0.6, postion[1], postion[2] + 0.55],
                "width": 0.6,
                "height": 1.2,
                "length": 0.1,
            },
            "rgba": [0.9, 0.9, 0.9, 1.0],
        }
        outer_left_obs = BoxObstacle(name="closet_outer_left", content_dict=outer_left)
        self.obstacles_3d.append(outer_left_obs)
        self.env.add_obstacle(outer_left_obs)

        outer_top = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [postion[0], postion[1], postion[2] + 1.1],
                "width": 0.6,
                "height": 0.1,
                "length": 1.2,
            },
            "rgba": [0.9, 0.9, 0.9, 1.0],
        }
        outer_top_obs = BoxObstacle(name="closet_outer_top", content_dict=outer_top)
        self.obstacles_3d.append(outer_top_obs)
        self.env.add_obstacle(outer_top_obs)

        # inner shelves
        for i in range(3):
            shelf = {
                "type": "box",
                "movable": False,
                "geometry": {
                    "position": [postion[0], postion[1], postion[2] + 0.35 + i * 0.25],
                    "width": 0.55,
                    "height": 0.05,
                    "length": 1.15,
                },
                "rgba": [0.8, 0.8, 0.8, 1.0],
            }
            shelf_obs = BoxObstacle(name=f"closet_shelf_{i}", content_dict=shelf)
            self.obstacles_3d.append(shelf_obs)
            self.env.add_obstacle(shelf_obs)

    def _create_walls(self):
        """Create room walls as box obstacles."""
        wall_thickness = 0.05
        wall_height = 0.5
        room_bounds = {"x_min": -0.5, "x_max": 5.0, "y_min": -0.5, "y_max": 6.0}

        # Left wall
        left_wall = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [
                    (room_bounds["y_min"] + room_bounds["y_max"]) / 2,
                    room_bounds["x_min"],
                    wall_height / 2,
                ],
                "width": wall_thickness,
                "height": wall_height,
                "length": room_bounds["y_max"] - room_bounds["y_min"],
            },
            "rgba": [0.7, 0.7, 0.7, 1.0],
        }
        left_wall_obs = BoxObstacle(name="left_wall", content_dict=left_wall)
        self.obstacles_3d.append(left_wall_obs)
        self.env.add_obstacle(left_wall_obs)

        # right wall
        right_wall = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [
                    (room_bounds["y_min"] + room_bounds["y_max"]) / 2,
                    room_bounds["x_max"],
                    wall_height / 2,
                ],
                "width": wall_thickness,
                "height": wall_height,
                "length": room_bounds["y_max"] - room_bounds["y_min"],
            },
            "rgba": [0.7, 0.7, 0.7, 1.0],
        }
        right_wall_obs = BoxObstacle(name="right_wall", content_dict=right_wall)
        self.obstacles_3d.append(right_wall_obs)
        self.env.add_obstacle(right_wall_obs)

        # front wall
        front_wall = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [
                    room_bounds["y_min"],
                    (room_bounds["x_min"] + room_bounds["x_max"]) / 2,
                    wall_height / 2,
                ],
                "width": room_bounds["x_max"] - room_bounds["x_min"],
                "height": wall_height,
                "length": wall_thickness,
            },
            "rgba": [0.7, 0.7, 0.7, 1.0],
        }
        front_wall_obs = BoxObstacle(name="front_wall", content_dict=front_wall)
        self.obstacles_3d.append(front_wall_obs)
        self.env.add_obstacle(front_wall_obs)

        # back wall
        back_wall = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [
                    room_bounds["y_max"],
                    (room_bounds["x_min"] + room_bounds["x_max"]) / 2,
                    wall_height / 2,
                ],
                "width": room_bounds["x_max"] - room_bounds["x_min"],
                "height": wall_height,
                "length": wall_thickness,
            },
            "rgba": [0.7, 0.7, 0.7, 1.0],
        }
        back_wall_obs = BoxObstacle(name="back_wall", content_dict=back_wall)
        self.obstacles_3d.append(back_wall_obs)
        self.env.add_obstacle(back_wall_obs)

    def _create_football_obstacle(self, position):
        football = {
            "type": "sphere",
            "movable": True,
            "geometry": {
                "position": position,
                "radius": 0.11,
            },
            "rgba": [1.0, 1.0, 1.0, 1.0],
        }
        football_obs = SphereObstacle(name="football", content_dict=football)
        self.obstacles_3d.append(football_obs)
        self.env.add_obstacle(football_obs)

    def _create_toycar_obstacle(self, position):
        """simple box + cylinders"""
        # car body
        car_body = {
            "type": "box",
            "movable": False,
            "geometry": {"position": position, "width": 0.2, "height": 0.1, "length": 0.4},
            "rgba": [1.0, 0.0, 0.0, 1.0],
        }
        car_body_obs = BoxObstacle(name="toycar_body", content_dict=car_body)
        self.obstacles_3d.append(car_body_obs)
        self.env.add_obstacle(car_body_obs)

        # wheels
        wheel_offsets = [(-0.1, -0.15), (0.1, -0.15), (-0.1, 0.15), (0.1, 0.15)]
        for i, (dx, dy) in enumerate(wheel_offsets):
            wheel = {
                "type": "cylinder",
                "movable": False,
                "geometry": {
                    "position": [position[0] + dx, position[1] + dy, position[2] + 0.05],
                    "radius": 0.05,
                    "height": 0.1,
                    "orientation": [
                        0,
                        0,
                        float(np.sin((0.5 * np.pi) / 2)),
                        float(np.cos((0.5 * np.pi) / 2)),
                    ],
                },
                "rgba": [0.0, 0.0, 0.0, 1.0],
            }
            wheel_obs = CylinderObstacle(name=f"toycar_wheel_{i}", content_dict=wheel)
            self.obstacles_3d.append(wheel_obs)
            self.env.add_obstacle(wheel_obs)

    def get_graspable_goals(self):
        """
        Get list of graspable object goals from the environment.

        Returns:
            List of dicts with:
                - name: object identifier
                - position: 3D numpy array [x, y, z]
                - object_type: type of object (for planning params)
                - radius: object approximate radius for safety
        """
        goals = []

        # Chair
        if self.chair_marker:
            goals.append(
                {
                    "name": "chair",
                    "position": np.array(self.chair_marker),
                    "object_type": "furniture",
                    "radius": 0.5,  # Approximate object size
                }
            )

        # Closet
        if self.closet_marker:
            goals.append(
                {
                    "name": "closet",
                    "position": np.array(self.closet_marker),
                    "object_type": "furniture",
                    "radius": 0.6,
                }
            )

        return goals

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
                "name": getattr(obstacle, "name", getattr(obstacle, "_name", "obstacle")),
                "type": obstacle.type(),
            }

            # Get 3D position and project to 2D (x, y plane, ignoring z)
            position_3d = obstacle.position()
            obs_2d["position"] = np.array([position_3d[0], position_3d[1]])

            # Extract size parameters based on type
            if obstacle.type() == "sphere":
                obs_2d["radius"] = obstacle.radius()

            elif obstacle.type() == "box":
                obs_2d["width"] = obstacle.length()
                obs_2d["length"] = obstacle.width()
                obs_2d["bounding_radius"] = self.bounding_circle_radius(obstacle)

            elif obstacle.type() == "cylinder":
                obs_2d["radius"] = obstacle.radius()

            obstacles_2d.append(obs_2d)

        return obstacles_2d

    def bounding_circle_radius(self, object):
        """
        Calculate conservative bounding circle radius for an obstacle.
        Used for quick collision filtering.
        """
        if object.type() == "box":
            return np.sqrt(object.width() ** 2 + object.length() ** 2) / 2
        elif object.type() == "cylinder":
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
            hw = object.width() / 2 + self.r_robot  # half-width + robot radius
            hl = object.length() / 2 + self.r_robot  # half-length + robot radius
            corners = [
                center + np.array([hw, hl, 0]),  # top right
                center + np.array([hw, -hl, 0]),  # bottom right
                center + np.array([-hw, -hl, 0]),  # bottom left
                center + np.array([-hw, hl, 0]),  # top left
            ]
            return corners

        elif object.type() == "cylinder":
            points = []
            r = object.radius() + self.r_robot
            for angle in np.linspace(0, 2 * np.pi, num=10, endpoint=False):
                point = center + np.array([r * np.cos(angle), r * np.sin(angle), 0])
                points.append(point)
            return points

        else:
            raise ValueError(f"Unknown obstacle type: {object.type()}")

    def collision_check(self, object1, object2):
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
