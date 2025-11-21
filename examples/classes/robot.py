class BaseRobot():

    def __init__(self, policy, model):
        self.policy = policy
        self.state = None
        self.model = model

    def get_model(self):
        return self.model

    def update_state(self, state):
        self.state = state

    def move(self):
        action = self.policy.compute_action(self.state)
        return action

