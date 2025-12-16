import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle



class PlaygroundEnv():
#define a enviroment that ubdates the pybullet gym, while also supplying the RRT 
#with a 2D map of the environment with all the nessesairy data.
#output will be a dictionairy with dimensions of the playground plane and a list of obstacles and their positions plus dimensions


    def __init__(self, random = False, obstacle_count = 1, steps = 300, end_pos = [9,4.5,0]
                 ,robot = GenericUrdfReacher(urdf="mobilePanda_with_gripper.urdf", mode="vel")
                 ,render = False):
        self.random = random
        self.obstacle_count = obstacle_count
        self.steps = steps
        self.obstacles_3d = []
        self.env = UrdfEnv(
            dt=0.01, robots=[robot], render=render, num_sub_steps= steps
        )
        self.end_pos = end_pos
        self.setup_obstacles(self.env)




    def setup_obstacles(self, env: UrdfEnv):
        obstacles = []
        if self.random:
            for i in range(self.obstacle_count):
            
                # Randomly generate obstacle parameters
                box_obst_dict = {
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
                "rgba": [0.8, 0.2, 0.2, 1.0] 
                }
                box_obst = BoxObstacle(name=f"box_{i}", content_dict=box_obst_dict)
                self.obstacles_3d.append(box_obst)
                env.add_obstacle(box_obst)
        
        else:
        
        
        
                # Add a box obstacle
                box_obst_dict_1 = {
                    "type": "box",
                    "movable": False,
                    "geometry": {
                        "position": [3, 2, 0.15],
                        "orientation": [0.5, 0, 0, 0.707],
                        "width": 0.8,
                        "height": 2,
                        "length": 1.8,
                    },
                    "rgba": [0.8, 0.2, 0.2, 1.0] 
}
            
                box_obst = BoxObstacle(name="box_1", content_dict=box_obst_dict_1)
                self.obstacles_3d.append(box_obst)
                env.add_obstacle(box_obst)

                #Lego pile
                random_list = np.random.uniform(-0.5,0.5,100)
                clear = True
                for i in range(7):
                    #Only add lego if it doesn't overlap with existing obstacles
                    while clear:

                        lego = {
                            "type": "box",
                            "movable": True,
                            "geometry": {
                                "position": [float(7 + np.random.choice(random_list)),
                                            float(4 + np.random.choice(random_list)),
                                            0.05
                                                ],
                                "orientation": [float(np.random.choice(random_list)), 0, 0, 1],
                                "width": 0.2,
                                "height": 0.1,
                                "length": 0.08,
                            },   
                            "rgba": [0.4, 0.2, 0.2, 1.0] 
        }               
                        
                        
                        box_obst = BoxObstacle(name=f"lego_{i}", content_dict=lego)
                        
                        #check if both bounding circles overlap.
                        clear = False
                        for obst in self.obstacles_3d:
                            dist = np.linalg.norm(np.array(lego["geometry"]["position"]) - np.array(obst.position()))
                            tresh = self.bounding_circle_radius(obst) + self.bounding_circle_radius(box_obst)
                            if dist < tresh:
                                clear = True
                                continue

                        if not clear:
                            self.obstacles_3d.append(box_obst)
                            env.add_obstacle(box_obst)
                            clear = True
                            break



                # Duplo wall
                for j in range(3):
                    for i in range(3-j):
                        duplo = {
                            "type": "box",
                            "movable": False,
                            "geometry": {
                                "position": [6.0,
                                            1 + i*0.2 + j*0.1,
                                            0.1 + j*0.2
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
                        env.add_obstacle(box_obst)



    #2D projection of obstacles for path planning
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
            # use getattr with a simple fallback to avoid AttributeError
            obs_2d = {
                'name': getattr(obstacle, 'name', getattr(obstacle, '_name', 'obstacle')),
                'type': obstacle.type(),
            }
            
            # Get 3D position
            position_3d = obstacle.position()
            
            # Project to 2D (x, y plane, ignoring z)
            obs_2d['position'] = np.array([position_3d[0], position_3d[1]])
            
            # Extract size parameters based on type
            if obstacle.type() == 'sphere':
                obs_2d['radius'] = obstacle.radius()
                
            elif obstacle.type() == 'box':

                obs_2d['width'] = obstacle.width()
                obs_2d['length'] = obstacle.length()
                # Calculate bounding circle radius for conservative collision checking
                obs_2d['bounding_radius'] = self.bounding_circle_radius(obstacle)
                
            elif obstacle.type() == 'cylinder':
                obs_2d['radius'] = obstacle.radius()
            
            obstacles_2d.append(obs_2d)
        
        # print(obstacles_2d)
        return obstacles_2d
    
    
    def bounding_circle_radius(self, object):
        if object.type() == 'box':
            return np.sqrt(object.width()**2 + object.length()**2) / 2
        
        else:
            raise ValueError(f"Not box type: {object.type()}")
