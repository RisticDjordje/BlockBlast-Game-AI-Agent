import copy
import os
import random
from datetime import datetime


class BlockGameState:
    """Core game logic for the block placement game, separated from visualization."""

    # Define colors for the shapes
    COLORS = [
        (255, 191, 0),  # Yellow
        (255, 143, 0),  # Orange
        (252, 48, 28),  # Red
        (112, 199, 48),  # Green
        (62, 181, 208),  # Light Blue
        (35, 64, 143),  # Dark Blue
    ]

    # Define shape forms
    FORMS = [
        [[[1, 1], [1, 1]]],  # shape 1 - 2x2 square
        [[[1, 1, 1], [1, 1, 1]], [[1, 1], [1, 1], [1, 1]]],  # shape 2 - 3x2 rectangle
        [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],  # shape 3 - 3x3 square
        [  # shape 4 - L shape (3 length)
            [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
            [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
            [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 1], [1, 1, 1]],
        ],
        [  # shape 5 - L (2 with 3 length)
            [[1, 1, 1], [1, 0, 0]],
            [[1, 1, 1], [0, 0, 1]],
            [[0, 0, 1], [1, 1, 1]],
            [[1, 0, 0], [1, 1, 1]],
            [[1, 0], [1, 0], [1, 1]],
            [[0, 1], [0, 1], [1, 1]],
            [[1, 1], [0, 1], [0, 1]],
            [[1, 1], [1, 0], [1, 0]],
        ],
        [  # shape 6 - Z shape
            [[0, 1, 1], [1, 1, 0]],
            [[1, 1, 0], [0, 1, 1]],
            [[1, 0], [1, 1], [0, 1]],
            [[0, 1], [1, 1], [1, 0]],
        ],
        [  # shape 7 - T shape
            [[0, 1, 0], [1, 1, 1]],
            [[1, 0], [1, 1], [1, 0]],
            [[1, 1, 1], [0, 1, 0]],
            [[0, 1], [1, 1], [0, 1]],
        ],
        [[[1, 1]], [[1], [1]]],  # shape 8 - 2x1 rectangle
        [[[1, 1, 1]], [[1], [1], [1]]],  # shape 9 - 3x1 rectangle
        [  # shape 10 - S shape
            [[1, 0], [1, 1]],
            [[1, 1], [0, 1]],
            [[1, 1], [1, 0]],
            [[0, 1], [1, 1]],
        ],
        [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],  # shape 11 - 4x1 rectangle
        [[[1, 1, 1, 1, 1]], [[1], [1], [1], [1], [1]]],  # shape 12 - 5x1 rectangle
        [
            [[1, 0], [1, 1]],  # shape 13 - 2*2 L shape
            [[0, 1], [1, 1]],
            [[1, 1], [1, 0]],
            [[1, 1], [0, 1]],
        ],
    ]

    def __init__(self):
        # Initialize an empty 8x8 grid
        self.grid = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)]
        self.score = 0
        self.displayed_score = 0  # For visualization smoothing
        self.highest_score = self.get_high_score()
        self.start_time = datetime.now()

        # Game state tracking
        self.game_over = False
        self.combo_streak = False
        self.combos = [["COMBO 0"], 0]

        # Combo tracking
        self.placements_without_clear = 0  # Count placements without clearing lines
        self.MAX_COMBO_STREAK = (
            3  # Combo resets after this many placements without clears
        )

        # Generate initial shapes
        self.current_shapes = self.generate_valid_shapes()

        # Track changes for reward calculation
        self.last_action_score = 0
        self.last_lines_cleared = 0

        # Define combo naming scheme
        self.combo_names = {
            1: "",
            2: "DOUBLE ",
            3: "TRIPLE ",
            4: "QUAD ",
            5: "PENTA ",
            6: "HEXA ",
        }

    class Shape:
        """Represents a game piece with a form and color."""

        def __init__(self, form_data=None):
            """
            Initialize a shape with a form and random color.

            Args:
                form_data: Either a tuple [form_index, variant_index] or -1 for a 1x1 shape
            """
            try:
                if form_data != -1:
                    self.form = BlockGameState.FORMS[form_data[0]][form_data[1]]
                else:
                    # Handle special case for 1x1 shape
                    self.form = [[1]]

                self.color = random.choice(BlockGameState.COLORS)
            except Exception:
                # Fallback to ensure we always have a valid shape
                self.form = [[1]]
                self.color = random.choice(BlockGameState.COLORS)

    def can_place_shape(self, shape, row, col):
        """Check if a shape can be placed at a specific position on the grid."""
        if not hasattr(shape, "form") or not shape.form:
            return False

        size = [len(shape.form), len(shape.form[0])]

        # Early boundary check
        if row < 0 or col < 0 or row + size[0] > 8 or col + size[1] > 8:
            return False

        # Check overlap
        for i in range(size[0]):
            for j in range(size[1]):
                if shape.form[i][j] and self.grid[row + i][col + j]:
                    return False

        return True

    def find_best_placement_for_shape(self, shape):
        """Find the best position to place a shape that might clear lines."""
        if not hasattr(shape, "form") or not shape.form:
            return None

        size = [len(shape.form), len(shape.form[0])]
        best_position = None
        max_potential_clears = -1

        for row in range(8 - size[0] + 1):
            for col in range(8 - size[1] + 1):
                if not self.can_place_shape(shape, row, col):
                    continue

                # Simulate placing the shape here
                temp_grid = [row[:] for row in self.grid]  # Create a deep copy
                for i in range(size[0]):
                    for j in range(size[1]):
                        if shape.form[i][j]:
                            temp_grid[row + i][
                                col + j
                            ] = 1  # Just mark as filled, color doesn't matter

                # Count potential lines cleared
                potential_clears = 0
                for r in range(8):
                    if all(temp_grid[r]):
                        potential_clears += 1

                for c in range(8):
                    if all(temp_grid[r][c] for r in range(8)):
                        potential_clears += 1

                if potential_clears > max_potential_clears:
                    max_potential_clears = potential_clears
                    best_position = (row, col)

        return best_position

    def simulate_placement(self, shape, position):
        """Simulate placing a shape on the grid and return the new grid after clearing lines."""
        if not position or not hasattr(shape, "form") or not shape.form:
            return self.grid

        # Create a deep copy of the grid
        new_grid = [row[:] for row in self.grid]

        # Place the shape
        row, col = position
        size = [len(shape.form), len(shape.form[0])]
        for i in range(size[0]):
            for j in range(size[1]):
                if shape.form[i][j]:
                    new_grid[row + i][col + j] = 1  # Mark as filled

        # Clear completed rows
        for r in range(8):
            if all(new_grid[r]):
                for c in range(8):
                    new_grid[r][c] = 0

        # Clear completed columns
        for c in range(8):
            if all(new_grid[r][c] for r in range(8)):
                for r in range(8):
                    new_grid[r][c] = 0

        return new_grid

    def generate_valid_shapes(self):
        """Generate three shapes that can each be placed sequentially on the board."""
        # Make copies to avoid modifying the original
        remaining_forms = [i for i in range(len(self.FORMS))]

        next_shapes = [0, 0, 0]  # Initialize with placeholders
        current_grid = [row[:] for row in self.grid]

        for shape_index in range(3):
            placeable_shape = None
            attempts = 0

            # Try to find a shape that can be placed
            while placeable_shape is None and attempts < 100:
                if not remaining_forms:
                    # If we've exhausted all forms, start over
                    remaining_forms = [i for i in range(len(self.FORMS))]

                # Pick a random form
                form_index = random.choice(remaining_forms)
                remaining_forms.remove(form_index)

                # Try each variant of this form
                for variant_index in range(len(self.FORMS[form_index])):
                    test_shape = self.Shape([form_index, variant_index])

                    # Check if this shape can be placed anywhere
                    for row in range(8 - len(test_shape.form) + 1):
                        for col in range(8 - len(test_shape.form[0]) + 1):
                            if self.can_place_shape(test_shape, row, col):
                                placeable_shape = test_shape
                                # Simulate placing this shape and clearing any lines
                                current_grid = self.simulate_placement(
                                    test_shape, (row, col)
                                )
                                break
                        if placeable_shape:
                            break
                    if placeable_shape:
                        break

                attempts += 1

            # If we couldn't find a placeable shape after many attempts, use a simple shape
            if placeable_shape is None:
                simple_shapes_indexes = [
                    [0, 0],  # 2x2 square
                    [7, 0],  # 2x1 rectangle
                    [8, 0],  # 3x1 rectangle
                ]

                for shape_indices in simple_shapes_indexes:
                    try:
                        simple_shape = self.Shape([shape_indices[0], shape_indices[1]])

                        # Check if this simple shape can be placed anywhere
                        can_place = False
                        for row in range(8 - len(simple_shape.form) + 1):
                            for col in range(8 - len(simple_shape.form[0]) + 1):
                                if self.can_place_shape(simple_shape, row, col):
                                    can_place = True
                                    current_grid = self.simulate_placement(
                                        simple_shape, (row, col)
                                    )
                                    break
                            if can_place:
                                break

                        if can_place:
                            placeable_shape = simple_shape
                            break
                    except Exception:
                        continue

            # If we still couldn't find a shape (board is nearly full), create a 1x1 shape
            if placeable_shape is None:
                try:
                    # Create a custom 1x1 shape as a last resort
                    one_by_one = self.Shape(-1)  # Using -1 as a special signal
                    placeable_shape = one_by_one
                except Exception:
                    # If all else fails, keep the placeholder
                    continue

            # Update the shape in our list
            next_shapes[shape_index] = placeable_shape

        return next_shapes

    def is_valid_placement(self, shape_idx, row, col):
        """Check if a shape can be placed at the specified position."""
        # Validate shape index
        if (
            shape_idx < 0
            or shape_idx >= len(self.current_shapes)
            or not self.current_shapes[shape_idx]
        ):
            return False

        # Get the shape
        shape = self.current_shapes[shape_idx]
        if not hasattr(shape, "form"):
            return False

        size = [len(shape.form), len(shape.form[0])]

        # Early boundary check (to avoid unnecessary iterations)
        if row < 0 or col < 0 or row + size[0] > 8 or col + size[1] > 8:
            return False

        # Check for overlaps
        for i in range(size[0]):
            for j in range(size[1]):
                if shape.form[i][j]:
                    if self.grid[row + i][col + j]:
                        return False

        return True

    def get_valid_actions(self):
        """Return all valid (shape_idx, row, col) combinations."""
        valid_actions = []

        for shape_idx in range(len(self.current_shapes)):
            # Check if the shape is not used (not equal to 0)
            if not self.current_shapes[shape_idx] or not hasattr(
                self.current_shapes[shape_idx], "form"
            ):
                continue

            shape = self.current_shapes[shape_idx]
            size = [len(shape.form), len(shape.form[0])]

            for row in range(8 - size[0] + 1):
                for col in range(8 - size[1] + 1):
                    if self.is_valid_placement(shape_idx, row, col):
                        valid_actions.append((shape_idx, row, col))

        return valid_actions

    def place_shape(self, shape_idx, row, col):
        """Attempt to place a shape on the grid and update the game state if it is successful.
        Returns is_valid_placement, new shapes generated."""
        if not self.is_valid_placement(shape_idx, row, col):
            return False, False

        is_valid_placement = True

        # Get the shape and its dimensions
        shape = self.current_shapes[shape_idx]
        size = [len(shape.form), len(shape.form[0])]
        
        volume = 0
        # Place the shape on the grid
        for i in range(size[0]):
            for j in range(size[1]):
                if shape.form[i][j]:
                    volume += 1
                    self.grid[row + i][col + j] = shape.color

        # Remove the used shape
        self.current_shapes[shape_idx] = 0

        # Update the grid (clear lines, calculate score)
        lines_cleared = self.update_grid(volume)

        # Update combo streak tracking
        if lines_cleared > 0:
            # Reset the counter when lines are cleared
            self.placements_without_clear = 0
        else:
            # Increment the counter when no lines are cleared
            self.placements_without_clear += 1

            # Reset combo if too many placements without clearing
            if self.placements_without_clear >= self.MAX_COMBO_STREAK:
                self.combo_streak = False
                self.combos[1] = 0
                self.combos[0][-1] = "COMBO 0"
                self.placements_without_clear = 0

        # Generate new shapes if all current shapes are used
        new_shapes_generated = False
        if all(shape == 0 for shape in self.current_shapes):
            self.current_shapes = self.generate_valid_shapes()
            new_shapes_generated = True

        # Check if the game is over
        self.check_game_over()

        return is_valid_placement, new_shapes_generated

    def update_grid(self, volume_placed=0):
        """Clear completed rows/columns and update score."""
        self.last_lines_cleared = 0
        score_before = self.score

        # Add block volume to score
        self.score += volume_placed

        # Find rows and columns to delete
        rows_to_delete = []
        cols_to_delete = []

        for i in range(8):
            if all(self.grid[i][j] for j in range(8)):
                rows_to_delete.append(i)

            if all(self.grid[j][i] for j in range(8)):
                cols_to_delete.append(i)

        # Clear the rows
        for row in rows_to_delete:
            for i in range(8):
                self.grid[row][i] = 0

        # Clear the columns
        for col in cols_to_delete:
            for i in range(8):
                self.grid[i][col] = 0

        # Check for all clear bonus
        all_clear = True
        for i in range(8):
            for j in range(8):
                if self.grid[i][j]:
                    all_clear = False
                    break
            if not all_clear:
                break

        # Update score
        lines_cleared = len(rows_to_delete) + len(cols_to_delete)
        self.last_lines_cleared = lines_cleared

        if lines_cleared:
            self.combos[0] = []

            # Calculate bonus based on combos and number of lines cleared
            bonus = lines_cleared * 10 * (self.combos[1] + 1)
            if lines_cleared > 2:
                bonus *= lines_cleared - 1

            # Add combo information
            # combo = self.combo_names.get(lines_cleared, "MULTI ") + f"CLEAR"

            combo = f"{lines_cleared} CLEAR"
            self.combos[0].insert(-1, combo)

            # Add all clear bonus
            if all_clear:
                bonus += 300
                self.combos[0].insert(-1, "ALL CLEAR")

            # Limit combo history
            # self.combos[0] = self.combos[0][-8:]

            # Update combo count - increase by the number of rows and columns cleared
            self.combos[1] += 1 # lines_cleared
            self.combos[0][-1] = f"COMBO {self.combos[1]}"
            self.combo_streak = True

            # Update score
            self.score += bonus
        else:
            self.combos[0] = [f"COMBO {self.combos[1]}"]

        # Track score change for reward calculation
        self.last_action_score = self.score - score_before
        
        # Update highest score
        if self.score > self.highest_score:
            self.highest_score = self.score

        return lines_cleared

    def check_game_over(self):
        """Check if there are any valid moves left."""
        self.game_over = True

        for shape_idx in range(len(self.current_shapes)):
            shape = self.current_shapes[shape_idx]
            if not shape or not hasattr(shape, "form"):
                continue

            size = [len(shape.form), len(shape.form[0])]

            # Only loop through valid starting positions based on shape size
            for row in range(8 - size[0] + 1):
                for col in range(8 - size[1] + 1):
                    # Quick check - if any position works, game isn't over
                    valid = True
                    for i in range(size[0]):
                        for j in range(size[1]):
                            if shape.form[i][j] and self.grid[row + i][col + j]:
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        self.game_over = False
                        return

    def get_state(self):
        """Return a dictionary with the current game state."""
        return {
            "grid": copy.deepcopy(self.grid),
            "available_shapes": self.current_shapes,
            "score": self.score,
            "game_over": self.game_over,
            "combo_streak": self.combo_streak,
            "combos": copy.deepcopy(self.combos),
            "placements_without_clear": self.placements_without_clear,
        }

    def get_normalized_state(self):
        """Return a normalized representation of the state for RL."""
        # Create binary grid (1 for filled, 0 for empty)
        binary_grid = [[1 if cell else 0 for cell in row] for row in self.grid]

        # Encode the shapes as binary matrices
        shapes_encoding = []
        for shape in self.current_shapes:
            if shape and hasattr(shape, "form"):
                shapes_encoding.append(shape.form)
            else:
                # Add an empty shape representation
                shapes_encoding.append([[0]])

        return {
            "grid": binary_grid,
            "shapes": shapes_encoding,
            "score": self.score,
            "combo": self.combos[1],
            "placements_without_clear": self.placements_without_clear,
        }

    def get_high_score(self):
        """Get the high score from file."""
        if not os.path.exists("high_score.txt"):
            self.save_score(0)
            return 0

        try:
            with open("high_score.txt", "r") as f:
                score = f.read().strip()
                return int(score) if score.isdigit() else 0
        except (FileNotFoundError, ValueError):
            self.save_score(0)
            return 0

    def save_score(self, score):
        """Save a score to the high score file."""
        with open("high_score.txt", "w") as file:
            file.write(str(score))

    def get_time(self):
        """Get the elapsed time since the game started."""
        if not self.start_time:
            return 0

        elapsed_time = datetime.now() - self.start_time
        return int(elapsed_time.total_seconds())

    def reset(self):
        """Reset the game state."""
        if self.score >= self.highest_score:
            self.save_score(self.score)

        self.grid = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)]
        self.score = 0
        self.displayed_score = 0
        self.game_over = False
        self.combo_streak = False
        self.combos = [["COMBO 0"], 0]
        self.placements_without_clear = 0
        self.current_shapes = self.generate_valid_shapes()
        self.last_action_score = 0
        self.last_lines_cleared = 0
        self.highest_score = self.get_high_score()
        self.start_time = datetime.now()
        
