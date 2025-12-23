import numpy as np

class ArmController:
    """
    A simple controller that ignores external targets and 
    just drives the arm to a 'Straight Up' (Candle) configuration.
    """
    
    def __init__(self, robot_id=None, kp=10.5, max_vel=0.5):
        # We accept robot_id to stay compatible with main.py, but we don't need it here.
        self.kp = kp
        self.max_vel = max_vel
        
        # Target: All zeros = Straight Up for Panda
        self.target_configuration = np.zeros(7)

    def get_target_velocities(self, current_joint_angles, target_pos_xyz=None):
        """
        Calculates velocities to move arm to the 'Up' position.
        The 'target_pos_xyz' argument is ignored for now.
        """
        current_joints = np.array(current_joint_angles)
        
        # 1. Calculate Error (Target - Current)
        error = self.target_configuration - current_joints
        
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