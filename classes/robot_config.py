"""
Panda robot configuration constants.
Centralized location for all robot-specific parameters.
"""

import numpy as np


class PandaConfig:
    """Configuration constants for the Panda robot arm on mobile base."""

    # Robot identification
    ROBOT_NAME = "robot_0"

    # Joint indices in URDF
    BASE_JOINT_INDICES = [0, 1, 2]  # x, y, theta (mobile base)
    ARM_JOINT_INDICES = [4, 5, 6, 7, 8, 9, 10]  # Panda arm (j1-j7)
    GRIPPER_JOINT_INDICES = [11, 12]  # Gripper fingers

    # Link indices
    BASE_LINK_INDEX = -1
    EE_LINK_INDEX = 14

    # Degrees of freedom
    BASE_DOF = 3
    ARM_DOF = 7
    GRIPPER_DOF = 2
    TOTAL_DOF = 12  # Base + Arm + Gripper

    # Action vector slices
    BASE_SLICE = slice(0, 3)  # action[0:3] for base velocities
    ARM_SLICE = slice(3, 10)  # action[3:10] for arm velocities
    GRIPPER_SLICE = slice(10, 12)  # action[10:12] for gripper

    # Observation slices
    BASE_OBS_SLICE = slice(0, 3)  # state[0:3] for base position
    ARM_OBS_SLICE = slice(3, 10)  # state[3:10] for arm joints

    # Panda arm joint limits (radians)
    ARM_JOINT_LIMITS_LOWER = np.array([-5.89, -3.76, -2.89, -3.07, -2.89, -0.01, -2.89])
    ARM_JOINT_LIMITS_UPPER = np.array([5.89, 3.76, 2.89, -0.06, 2.89, 3.75, 2.89])

    # Candle configuration (straight up, safe for navigation)
    CANDLE_CONFIGURATION = np.zeros(7)

    # Physical dimensions
    BASE_RADIUS = 0.001
