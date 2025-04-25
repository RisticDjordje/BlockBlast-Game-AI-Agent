import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .game_state import BlockGameState


class BlockGameEnv(gym.Env):
    """
    Environment wrapper for the block placement game.
    Implements a gym-like interface for RL agents.
    """

    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Initialize game state
        self.game_state = BlockGameState()

        # Define action and observation spaces
        # Action space: (shape_idx (0-2), row (0-7), col (0-7))
        # We'll use a flat discrete action space (0-191) for compatibility
        self.action_space = spaces.Discrete(3 * 8 * 8)  # 3 pieces × 64 positions

        # Observation space: grid, shapes, score
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.int8),
                "shapes": spaces.Box(
                    low=0, high=1, shape=(3, 5, 5), dtype=np.int8
                ),  # Max shape size is 5×5
                "score": spaces.Box(
                    low=0, high=float("inf"), shape=(1,), dtype=np.float32
                ),
                "combo": spaces.Box(
                    low=0, high=float("inf"), shape=(1,), dtype=np.int8
                ),
            }
        )

        # Set up rendering
        self.render_mode = render_mode
        self.renderer = None

        if self.render_mode == "human":
            # Import here to avoid PyGame dependency if not rendering
            from .game_renderer import BlockGameRenderer

            self.renderer = BlockGameRenderer(self.game_state)

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        # Reset the game state
        self.game_state.reset()

        # Get the initial observation
        observation = self._get_observation()

        # Render if in human mode
        if self.render_mode == "human" and self.renderer:
            self.renderer.render()

        return observation, {}

    def step(self, action):
        """
        Apply an action to the environment.

        Args:
            action: A discrete action index (0-191)

        Returns:
            observation: The current state
            reward: The reward for the action
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated (time limit)
            info: Additional information
        """
        # Decode the action
        shape_idx, row, col = self._decode_action(action)

        # Track state before action for reward calculation
        old_game_over = self.game_state.game_over

        # Always attempt to apply the action, even if it's invalid
        # place_shape will internally check validity and only change state if valid
        valid_placement, new_shapes_generated = self.game_state.place_shape(
            shape_idx, row, col
        )

        # Reset chosen shape in renderer if new shapes were generated
        if new_shapes_generated and self.render_mode == "human" and self.renderer:
            self.renderer.chosen_shape = -1

        # Calculate reward based on whether the placement was valid
        reward = self._calculate_reward(valid_placement, old_game_over)

        # Get the new observation
        observation = self._get_observation()

        # Check if the episode has ended
        terminated = self.game_state.game_over
        truncated = False

        # Render if in human mode
        if self.render_mode == "human" and self.renderer:
            self.renderer.render()

        # Additional info
        info = {
            "valid_placement": valid_placement,
            "score": self.game_state.score,
            "lines_cleared": self.game_state.last_lines_cleared,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human" and self.renderer:
            self.renderer.render()
        elif self.render_mode == "rgb_array" and self.renderer:
            return self.renderer.get_rgb_array()

    def close(self):
        """Clean up resources and check for high score update."""
        if self.renderer:
            self.renderer.close()
        # Optionally, you can check for high score here if needed

    def _decode_action(self, action):
        """Convert a flat action index to (shape_idx, row, col)."""
        assert 0 <= action < 3 * 8 * 8, f"Action out of range: {action}"

        shape_idx = action // 64
        position = action % 64
        row = position // 8
        col = position % 8

        return shape_idx, row, col

    def _encode_action(self, shape_idx, row, col):
        """Convert (shape_idx, row, col) to a flat action index."""
        return shape_idx * 64 + row * 8 + col

    def _get_observation(self):
        """Convert game state to observation format."""
        # Get the normalized state
        state = self.game_state.get_normalized_state()

        # Convert grid to numpy array
        grid = np.array(state["grid"], dtype=np.int8)

        # Encode shapes as fixed-size arrays
        shapes = np.zeros((3, 5, 5), dtype=np.int8)
        for i, shape in enumerate(state["shapes"]):
            if shape:
                height, width = len(shape), len(shape[0])
                if height <= 5 and width <= 5:
                    shapes[i, :height, :width] = np.array(shape, dtype=np.int8)

        return {
            "grid": grid,
            "shapes": shapes,
            "score": np.array([state["score"]], dtype=np.float32),
            "combo": np.array([state["combo"]], dtype=np.int8),
        }

    def _calculate_reward(self, valid_placement, old_game_over):
        """
        Calculate the reward for the current action.

        """

        if not valid_placement:
            return -0.5  # Reduced penalty

        # Base reward for valid placement
        reward = 0.5

        # Points gained
        points_gained = self.game_state.last_action_score
        reward += 0.05 * points_gained  # Increased multiplier

        # Lines cleared - boost this component
        lines_cleared = self.game_state.last_lines_cleared
        reward += 1.0 * lines_cleared  # Increased multiplier

        # Game over penalty - make less severe
        if not old_game_over and self.game_state.game_over:
            return -3.0  # Less severe

        return reward

    def get_valid_actions(self):
        """Return a list of valid action indices."""
        valid_actions = []

        for shape_idx, row, col in self.game_state.get_valid_actions():
            valid_actions.append(self._encode_action(shape_idx, row, col))

        return valid_actions

    def action_masks(self):
        """
        Returns a boolean mask indicating which actions are valid.
        Required by MaskablePPO and can be used for DQN with action masking.

        Returns:
            np.ndarray: Boolean array of shape (action_space.n,) where
                       True indicates the action is valid.
        """
        valid_actions = self.get_valid_actions()
        mask = np.zeros(self.action_space.n, dtype=bool)

        # Set True for each valid action
        for action_idx in valid_actions:
            mask[action_idx] = True

        return mask
