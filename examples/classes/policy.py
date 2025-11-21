import numpy as np


class BasePolicy():

    def __init__(self):
        pass

    def compute_action(self, state):
        pass


class FeedbackPolicy(BasePolicy):

    def __init__(self, target_obstacle, forward_velocity, action_size, step=0):
        self.target_obstacle = target_obstacle
        self.forward_velocity = forward_velocity
        self.action_size = action_size
        self.step = step

    def compute_action(self, state):
        action = np.zeros(self.action_size)
        
        # Extract robot position from state
        robot_pos = np.array([0.0, 0.0, 0.0])
        if isinstance(state, dict) and 'position' in state:
            position_array = state['position']
            if len(position_array) >= 3:
                robot_pos[0] = position_array[0]  # x position
                robot_pos[1] = position_array[1]  # y position
                robot_pos[2] = position_array[2]  # z position
        
        # Calculate direction to target obstacle
        direction = self.target_obstacle[:2] - robot_pos[:2]  # 2D direction vector (x, y)
        distance = np.linalg.norm(direction)
        
        if distance > 0.05:  # If not very close yet
            # Normalize direction and scale by desired speed
            direction_normalized = direction / distance
            action[0] = direction_normalized[0] * self.forward_velocity  # x velocity
            action[1] = direction_normalized[1] * self.forward_velocity  # y velocity
        else:
            # Very close, just keep moving forward
            action[0] = self.forward_velocity
            action[1] = 0.0
        
        # Alternate gripper actions every 100 steps
        if (int(self.step / 100)) % 2 == 0:
            action[-1] = -0.01  # Close finger 2
            action[-2] = -0.01  # Close finger 1
        else:
            action[-1] = 0.01   # Open finger 2
            action[-2] = 0.01   # Open finger 1
        
        self.step += 1
        return action

