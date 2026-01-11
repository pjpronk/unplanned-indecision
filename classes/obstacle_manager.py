import numpy as np
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
import pybullet as p


class ObstacleManager:
    """
    ObstacleManager class that manages obstacles for the URDF environment.
    Provides methods to get 3D and 2D obstacle representations.
    """

    def __init__(self):
        """Initialize the environment with predefined obstacles."""
        self.obstacles_3d = []
        self._create_obstacles()

    def _create_obstacles(self):
        """Create all obstacles and store them in memory."""
        # Sphere obstacle
        sphere_obst_dict = {
            "type": "sphere",
            "movable": False,
            "geometry": {"position": [-0.5, 0.0, 0.2], "radius": 0.25},
            "rgba": [0.3, 0.5, 0.6, 1.0],
        }
        sphere_obst = SphereObstacle(name="sphere_1", content_dict=sphere_obst_dict)
        self.obstacles_3d.append(sphere_obst)

        # Box obstacle
        box_obst_dict = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [2.0, -0.5, 0.15],
                "width": 0.3,
                "height": 0.3,
                "length": 0.3,
            },
            "rgba": [0.8, 0.2, 0.2, 1.0],
        }
        box_obst = BoxObstacle(name="box_1", content_dict=box_obst_dict)
        self.obstacles_3d.append(box_obst)

        # Cylinder obstacle
        cylinder_obst_dict = {
            "type": "cylinder",
            "movable": False,
            "geometry": {
                "position": [1.5, 1.0, 0.3],
                "radius": 0.15,
                "height": 0.6,
            },
            "rgba": [0.2, 0.8, 0.2, 1.0],
        }
        cylinder_obst = CylinderObstacle(name="cylinder_1", content_dict=cylinder_obst_dict)
        self.obstacles_3d.append(cylinder_obst)

        # Shelf obstacle
        shelf_obst_dict = {
            "type": "box",
            "movable": False,
            "geometry": {
                "position": [2.5, -1.5, 0.6],  # Same X/Y as target, higher Z
                "width": 0.1,  # Wide enough to block top approach
                "length": 0.1,
                "height": 0.1,  # Thin like a shelf
            },
            "rgba": [0.4, 0.4, 0.4, 1.0],  # Dark Grey
        }
        shelf_obst = BoxObstacle(name="shelf_1", content_dict=shelf_obst_dict)
        self.obstacles_3d.append(shelf_obst)

    def add_target_visual(self, position, size=0.05, rgba=(1.0, 0.0, 1.0, 0.6)):
        """
        Spawns a *visual-only* pink cube in PyBullet (no collision shape).
        Returns the PyBullet body_id of the visual marker.
        """
        half = size / 2.0

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=rgba
        )

        # baseCollisionShapeIndex = -1  means no collision at all
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=-1,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

        return body_id

    def get_3d_obstacles(self):
        """
        Get list of 3D obstacle objects.

        Returns:
            list: List of obstacle objects (SphereObstacle, BoxObstacle, CylinderObstacle)
        """
        return self.obstacles_3d

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
            obs_2d = {
                "name": obstacle.name,
                "type": obstacle.type(),
            }

            # Get 3D position
            position_3d = obstacle.position()

            # Project to 2D (x, y plane, ignoring z)
            obs_2d["position"] = np.array([position_3d[0], position_3d[1]])

            # Extract size parameters based on type
            if obstacle.type() == "sphere":
                obs_2d["radius"] = obstacle.radius()

            elif obstacle.type() == "box":
                obs_2d["width"] = obstacle.width()
                obs_2d["length"] = obstacle.length()
                # Calculate bounding circle radius for conservative collision checking
                obs_2d["bounding_radius"] = (
                    np.sqrt(obstacle.width() ** 2 + obstacle.length() ** 2) / 2
                )

            elif obstacle.type() == "cylinder":
                obs_2d["radius"] = obstacle.radius()

            obstacles_2d.append(obs_2d)

        return obstacles_2d

    def add_to_urdf_env(self, urdf_env):
        """
        Add all obstacles to a URDF environment.

        Args:
            urdf_env: UrdfEnv instance to add obstacles to
        """
        for obstacle in self.obstacles_3d:
            urdf_env.add_obstacle(obstacle)
