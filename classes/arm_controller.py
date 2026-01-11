import numpy as np
import pybullet as p


class ArmController:
    """
    Simple P-controller for robotic arm.
    Can drive to candle (straight up) position or use IK to reach targets.
    """

    def __init__(self, robot_id, robot_config, dt, kp, max_vel):
        self.robot_id = robot_id
        self.config = robot_config
        self.dt = dt
        self.kp = kp
        self.max_vel = max_vel

        # Target: Candle configuration (straight up)
        self.target_configuration = self.config.candle_configuration

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

        # Ask PyBullet to solve IK for the arm
        ik_solution = p.calculateInverseKinematics(
            self.robot_id, self.config.ee_link_index, target_pos_3d
        )

        # Extract arm joints from IK solution
        target_arm_joints = np.array(ik_solution[self.config.arm_obs_slice])

        return target_arm_joints

    def get_target_velocities(self, current_joint_angles, target_pos_3d=None):
        """
        Calculates velocities to move arm to target position or candle position.

        Args:
            current_joint_angles: Current arm joint angles (7 values)
            target_pos_3d: Optional [x, y, z] target. If None, uses candle position.

        Returns:
            Joint velocities (7 values)
        """
        current_joints = np.array(current_joint_angles)

        # If no target provided, use candle (straight up) configuration
        if target_pos_3d is None:
            target_joints = self.target_configuration
        else:
            # Calculate the desired joint angles using IK
            target_joints = self._solve_inverse_kinematics(target_pos_3d)

        # Calculate Error (Target - Current)
        error = target_joints - current_joints

        # P-Controller (Velocity = Error * Gain)
        velocities = error * self.kp

        # Clip velocities for safety
        velocities = np.clip(velocities, -self.max_vel, self.max_vel)

        return velocities

    def has_arrived(self, current_joint_angles, target_pos_xyz=None, threshold=0.1) -> bool:
        """
        Checks if the arm is close to the target configuration.

        Args:
            current_joint_angles: Current arm joint angles (7 values)
            target_pos_xyz: Unused (for interface compatibility)
            threshold: Distance threshold in radians

        Returns:
            True if arm is within threshold of target configuration
        """
        current_joints = np.array(current_joint_angles)
        error = np.linalg.norm(self.target_configuration - current_joints)

        return error < threshold
