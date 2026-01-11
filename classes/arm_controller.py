import numpy as np
import pybullet as p


class ArmController:
    """
    Simple P-controller for robotic arm.
    Uses IK to reach targets.
    """

    def __init__(self, robot_id, robot_config, dt, kp, max_vel):
        self.robot_id = robot_id
        self.config = robot_config
        self.dt = dt
        self.kp = kp
        self.max_vel = max_vel

    def _solve_inverse_kinematics(self, target_pos_3d):
        """
        Computes the required joint angles to reach the target position.
        """
        ik_solution = p.calculateInverseKinematics(
            self.robot_id, self.config.ee_link_index, target_pos_3d
        )

        target_arm_joints = np.array(ik_solution[self.config.arm_obs_slice])
        return target_arm_joints

    def get_target_velocities(self, current_joint_angles, target_pos_3d):
        """
        Calculates velocities to move arm to target position.

        Args:
            current_joint_angles: Current arm joint angles
            target_pos_3d: Target 3D position (required)

        Returns:
            Array of joint velocities
        """
        current_joints = np.array(current_joint_angles)

        target_joints = self._solve_inverse_kinematics(target_pos_3d)
        error = target_joints - current_joints

        velocities = error * self.kp
        velocities = np.clip(velocities, -self.max_vel, self.max_vel)

        return velocities

    def get_joint_velocities(self, current_joint_angles, target_joint_angles):
        """
        Calculates velocities to move arm to target joint configuration.

        Args:
            current_joint_angles: Current arm joint angles
            target_joint_angles: Target joint configuration (e.g., candle_configuration)

        Returns:
            Array of joint velocities toward target configuration
        """
        current_joints = np.array(current_joint_angles)
        target_joints = np.array(target_joint_angles)
        error = target_joints - current_joints

        velocities = error * self.kp
        velocities = np.clip(velocities, -self.max_vel, self.max_vel)

        return velocities

    def has_arrived(self, current_joint_angles, target_pos_xyz, threshold=0.1):
        """
        Checks if the arm is close to the target position.

        Args:
            current_joint_angles: Current arm joint angles (7 values)
            target_pos_xyz: Target 3D position (required)
            threshold: Distance threshold in radians

        Returns:
            True if arm is within threshold of target configuration
        """
        current_joints = np.array(current_joint_angles)
        target_joints = self._solve_inverse_kinematics(target_pos_xyz)
        error = np.linalg.norm(target_joints - current_joints)

        return error < threshold

    def has_reached_joints(self, current_joint_angles, target_joint_angles, threshold=0.1):
        """
        Checks if the arm is close to the target joint configuration.

        Args:
            current_joint_angles: Current arm joint angles (7 values)
            target_joint_angles: Target joint configuration
            threshold: Distance threshold in radians

        Returns:
            True if arm is within threshold of target configuration
        """
        current_joints = np.array(current_joint_angles)
        target_joints = np.array(target_joint_angles)
        error = np.linalg.norm(target_joints - current_joints)

        return error < threshold
