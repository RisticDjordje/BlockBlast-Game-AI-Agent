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

        self.start_time = datetime.now()

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

    def _can_place_on(self, grid, shape, row, col):
        """Like can_place_shape, but tests against an explicit grid."""
        h, w = len(shape.form), len(shape.form[0])
        if row < 0 or col < 0 or row + h > 8 or col + w > 8:
            return False
        for i in range(h):
            for j in range(w):
                if shape.form[i][j] and grid[row + i][col + j]:
                    return False
        return True

    def _simulate_on(self, grid, shape, position):
        """Place & clear lines on a copy of grid, return the new grid."""
        new = [r[:] for r in grid]
        row, col = position
        h, w = len(shape.form), len(shape.form[0])
        # place
        for i in range(h):
            for j in range(w):
                if shape.form[i][j]:
                    new[row + i][col + j] = 1
        # clear full rows
        for r in range(8):
            if all(new[r][c] for c in range(8)):
                for c in range(8):
                    new[r][c] = 0
        # clear full cols
        for c in range(8):
            if all(new[r][c] for r in range(8)):
                for r in range(8):
                    new[r][c] = 0
        return new

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
        """
        Greedily pick 3 shapes so each one can be placed on the board
        updated by the previous placement.
        """
        remaining = list(range(len(self.FORMS)))
        next_shapes = []
        current_grid = [row[:] for row in self.grid]

        for _ in range(3):
            placed = False
            random.shuffle(remaining)
            for form_idx in remaining:
                for var_idx in range(len(self.FORMS[form_idx])):
                    shape = self.Shape([form_idx, var_idx])
                    # find _any_ valid spot on current_grid
                    for r in range(8 - len(shape.form) + 1):
                        for c in range(8 - len(shape.form[0]) + 1):
                            if self._can_place_on(current_grid, shape, r, c):
                                # commit to this spot
                                next_shapes.append(shape)
                                current_grid = self._simulate_on(
                                    current_grid, shape, (r, c)
                                )
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break
                if placed:
                    remaining.remove(form_idx)
                    break

            if not placed:
                # fallback to 1×1 if no multi-cell shape fits
                one_by_one = self.Shape(-1)
                # guaranteed to fit somewhere unless board is truly jammed
                for r in range(8):
                    for c in range(8):
                        if self._can_place_on(current_grid, one_by_one, r, c):
                            next_shapes.append(one_by_one)
                            current_grid = self._simulate_on(
                                current_grid, one_by_one, (r, c)
                            )
                            placed = True
                            break
                    if placed:
                        break
                # if even that fails, just append it anyway
                if not placed:
                    next_shapes.append(one_by_one)

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

        # Place the shape on the grid
        for i in range(size[0]):
            for j in range(size[1]):
                if shape.form[i][j]:
                    self.grid[row + i][col + j] = shape.color

        # Remove the used shape
        self.current_shapes[shape_idx] = 0

        # Update the grid (clear lines, calculate score)
        lines_cleared, all_clear = self.update_grid()

        # Update combo streak tracking
        if lines_cleared > 0:
            # Reset the counter when lines are cleared
            self.placements_without_clear = 0

            if not self.combo_streak:
                self.combo_streak = True
            else:
                self.combos[1] += 1
        else:
            # Increment the counter when no lines are cleared
            self.placements_without_clear += 1

            # Reset combo if too many placements without clearing
            if self.placements_without_clear >= self.MAX_COMBO_STREAK:
                self.combos[1] = 0
                self.combo_streak = False
                self.placements_without_clear = 0

        # Update score 
        self.update_score(lines_cleared, all_clear)

        # Generate new shapes if all current shapes are used
        new_shapes_generated = False
        if all(shape == 0 for shape in self.current_shapes):
            self.current_shapes = self.generate_valid_shapes()
            new_shapes_generated = True

        # Check if the game is over
        self.check_game_over()

        return is_valid_placement, new_shapes_generated

    def update_grid(self):
        """Clear completed rows/columns and update score."""
        
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
        
        # Return grid updation
        lines_cleared = len(rows_to_delete) + len(cols_to_delete)
        self.last_lines_cleared = lines_cleared

        return lines_cleared, all_clear

    def update_score(self, lines_cleared, all_clear):
        score_before = self.score
        self.combos[0] = [f"COMBO {self.combos[1]}"]

        if lines_cleared:
            # Calculate bonus based on combos and number of lines cleared
            bonus = lines_cleared * 10 * (self.combos[1] + 1)
            if lines_cleared > 2:
                bonus *= lines_cleared - 1

            # Add combo information
            combo = self.combo_names.get(lines_cleared, "MULTI ") + f"CLEAR +{bonus}"
            self.combos[0].insert(-1, combo)

            # Add all clear bonus
            if all_clear:
                bonus += 300
                self.combos[0].insert(-1, "ALL CLEAR +300")

            # Update score
            self.score += bonus
            if self.score > self.highest_score:
                self.highest_score = self.score

        # Track score change for reward calculation
        self.last_action_score = self.score - score_before

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
        if not os.path.exists("blockblast_game/high_score.txt"):
            self.save_score(0)
            return 0

        try:
            with open("blockblast_game/high_score.txt", "r") as f:
                score = f.read().strip()
                return int(score) if score.isdigit() else 0
        except (FileNotFoundError, ValueError):
            self.save_score(0)
            return 0

    def save_score(self, score):
        """Save a score to the high score file."""
        with open("blockblast_game/high_score.txt", "w") as file:
            file.write(str(score))

    def get_duration(self):
        """Get the duration of the game in seconds."""
        return int((datetime.now() - self.start_time).total_seconds())

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
