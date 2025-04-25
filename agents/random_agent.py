import random


class RandomAgent:
    """A simple random agent for demonstration purposes."""

    def __init__(self, env):
        self.env = env

    def predict(self, observation):
        """Choose a random valid action."""
        valid_actions = self.env.get_valid_actions()
        print(valid_actions)
        if not valid_actions:
            # If no valid actions, choose any action (env will handle the invalid action)
            return random.randint(0, 3 * 8 * 8 - 1), None
        return random.choice(valid_actions), None
