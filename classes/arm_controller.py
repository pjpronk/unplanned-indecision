import numpy as np
import pybullet as p

class ArmController:
    """
    A simple controller that ignores external targets and 
    just drives the arm to a 'Straight Up' (Candle) configuration.
    """
    
    def __init__(self, robot_id=None, dt=0.01, kp=10.5, max_vel=0.5, ee_idx=11):
        # We accept robot_id to stay compatible with main.py, but we don't need it here.
        self.robot_id = robot_id
        self.dt = dt
        self.kp = kp
        self.max_vel = max_vel
        self.ee_idx = ee_idx
        
        # Target: All zeros = Straight Up for Panda
        self.target_configuration = np.zeros(7)
    
    def _solve_inverse_kinematics(self, target_pos_3d):
        """
        Computes the required joint angles to reach the target position.
        Currently uses PyBullet's built-in numerical solver.
        
        Args:
            target_pos_3d: [x, y, z] coordinates of the target
            
        Returns:
            np.array: The target joint configuration (7 angles for the arm)
        """
        if self.robot_id is None:
            print("Warning: Robot ID not set. returning zeros.")
            return np.zeros(7)

        # 1. Ask PyBullet to solve IK
        # This returns all joint positions for the robot
        ik_solution = p.calculateInverseKinematics(
            self.robot_id, 
            self.ee_idx, 
            target_pos_3d
        )
        
        # 2. Extract the Arm Joints
        # Albert typically has: [x, y, theta, joint1, ... joint7, fingers...]
        # The IK solver returns a tuple of ALL movable joints.
        # We need to slice out the 7 arm joints. 
        # Usually, the arm starts at index 3 (after the 3 base DOFs) 
        # and ends at index 10.
        
        # Note: If this slicing is wrong, your arm will spasm. 
        # We will verify this in the next step.
        target_arm_joints = np.array(ik_solution[3:10])
        
        return target_arm_joints

    def get_target_velocities(self, current_joint_angles, target_pos_3d):
        """
        Calculates velocities to move arm to the 'Up' position.
        The 'target_pos_3d' argument is ignored for now.
        """

        current_joints = np.array(current_joint_angles)

        # 1. Calculate the DESIRED joint angles using your new function
        target_joints = self._solve_inverse_kinematics(target_pos_3d)
        
        # 1. Calculate Error (Target - Current)
        error = target_joints - current_joints
        
        # 2. P-Controller (Velocity = Error * Gain)
        velocities = error * self.kp
        
        # 3. Clip velocities for safety
        velocities = np.clip(velocities, -self.max_vel, self.max_vel)
        
        return velocities

    def has_arrived(self, current_joint_angles, target_pos_xyz=None, threshold=0.1):
        """
        Checks if the arm is close to the 'Up' configuration.
        """
        current_joints = np.array(current_joint_angles)
        error = np.linalg.norm(self.target_configuration - current_joints)
        
        return error < threshold