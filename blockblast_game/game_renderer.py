import pygame
import sys
import os
import numpy as np


class BlockGameRenderer:
    """Renderer for visualizing the block placement game."""

    def __init__(self, game_state, fps=60):
        """Initialize the renderer with a reference to the game state.

        Args:
            game_state: Instance of BlockGameState
            fps: Frames per second for rendering
        """
        # Store reference to the game state
        self.game_state = game_state

        # Initialize PyGame
        pygame.init()
        pygame.mixer.init()

        # Constants
        self.INIT_WIDTH = 1200
        self.INIT_HEIGHT = 800
        self.BACKGROUND_COLOR = (220, 220, 220)
        self.FPS = fps
        self.grid_line_width = 2

        # Set up display
        self.main_screen = pygame.display.set_mode(
            (self.INIT_WIDTH, self.INIT_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption("Block Blast Game")

        # Set up clock
        self.clock = pygame.time.Clock()

        # Determine asset paths
        module_dir = os.path.dirname(os.path.abspath(__file__))  # .../blockblast_game
        assets_dir = os.path.join(module_dir, "Assets")  # .../blockblast_game/Assets
        if not os.path.isdir(assets_dir):
            print(f"Warning: Assets directory not found at {assets_dir}")

        # Load clear sound
        sound_path = os.path.join(assets_dir, "clear_sf.wav")
        if not os.path.isfile(sound_path):
            print(f"Warning: Sound file not found at {sound_path}, using silent sound.")
            self.clear_sound = pygame.mixer.Sound(buffer=bytearray(10))
        else:
            self.clear_sound = pygame.mixer.Sound(sound_path)

        # Prepare font loading
        self.font_path = os.path.join(assets_dir, "LECO.ttf")
        if not os.path.isfile(self.font_path):
            print(
                f"Warning: Font file not found at {self.font_path}, using system font."
            )
            self.have_custom_font = False
        else:
            self.have_custom_font = True

        # Visualization state
        self.chosen_shape = -1
        self.agent_thinking = False
        self.highlight_position = None
        self.displayed_score = 0
        self.game_over_alpha = 0
        self.current_placement = None
        self.fade_alpha = 0
        self.transition_in = False
        self.transition_out = False
        self.TRANSITION_SPEED = 15
        self.debug_mode = False
        self._last_combo_count = len(self.game_state.combos[0])

    def make_font(self, size):
        """Helper to load custom or system font based on availability."""
        if self.have_custom_font:
            return pygame.font.Font(self.font_path, size)
        else:
            return pygame.font.SysFont("Arial", size)

    def render(self):
        """Render the current state of the game."""
        current = len(self.game_state.combos[0])
        if current > self._last_combo_count:
            self.clear_sound.play()
        self._last_combo_count = current

        self.main_screen.fill(self.BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_shapes()
        self.draw_score()
        self.draw_combos()

        if self.chosen_shape != -1:
            self.draw_cursor()
        if self.agent_thinking:
            self.draw_agent_thinking()
        if self.game_state.game_over:
            self.draw_game_over()
        if self.transition_in:
            self.fade_in()
        elif self.transition_out:
            self.fade_out()
        if self.debug_mode and self.current_placement:
            self.draw_debug_info()

        pygame.display.flip()
        self.clock.tick(self.FPS)

    def draw_debug_info(self):
        debug_font = pygame.font.SysFont("Arial", 16)
        y_pos = 10
        if self.current_placement:
            row, col = self.current_placement
            text_surface = debug_font.render(
                f"Placement: ({row},{col})", True, (255, 0, 0)
            )
            self.main_screen.blit(text_surface, (10, y_pos))
            y_pos += 20
        mouse_x, mouse_y = pygame.mouse.get_pos()
        text_surface = debug_font.render(
            f"Mouse: ({mouse_x},{mouse_y})", True, (255, 0, 0)
        )
        self.main_screen.blit(text_surface, (10, y_pos))
        y_pos += 20
        dims = self.calculate_grid_dimensions()
        info = f"Grid pos: ({int(dims['grid_pos_x'])},{int(dims['grid_pos_y'])}) cell: {dims['square_side']:.1f}"
        text_surface = debug_font.render(info, True, (255, 0, 0))
        self.main_screen.blit(text_surface, (10, y_pos))

    def calculate_grid_dimensions(self):
        width, height = self.main_screen.get_size()
        padding = height // 10
        grid_side = height - 2 * padding
        grid_side -= (grid_side % 8) - (self.grid_line_width * 7)
        x = (width - grid_side) / 2
        y = padding
        cell = (grid_side - self.grid_line_width * 7) / 8
        return {
            "grid_pos_x": x,
            "grid_pos_y": y,
            "grid_side": grid_side,
            "square_side": cell,
            "grid_padding": padding,
        }

    def draw_grid(self):
        dims = self.calculate_grid_dimensions()
        grid = pygame.Surface((dims["grid_side"], dims["grid_side"]))
        grid.fill((255, 255, 255))
        # outline
        rect = pygame.Rect(
            dims["grid_pos_x"] - 2,
            dims["grid_pos_y"] - 2,
            dims["grid_side"] + 4,
            dims["grid_side"] + 4,
        )
        pygame.draw.rect(self.main_screen, (0, 0, 0), rect)
        # lines
        step = (dims["grid_side"] - self.grid_line_width * 7) / 8
        pos = step
        for _ in range(7):
            pygame.draw.line(
                grid,
                (200, 200, 200),
                (0, pos),
                (dims["grid_side"], pos),
                self.grid_line_width,
            )
            pygame.draw.line(
                grid,
                (200, 200, 200),
                (pos, 0),
                (pos, dims["grid_side"]),
                self.grid_line_width,
            )
            pos += step + self.grid_line_width
        # cells
        for i in range(8):
            for j in range(8):
                color = self.game_state.grid[i][j]
                if color:
                    x = j * (dims["square_side"] + self.grid_line_width)
                    y = i * (dims["square_side"] + self.grid_line_width)
                    bg = pygame.Rect(
                        x - 2, y - 2, dims["square_side"] + 4, dims["square_side"] + 4
                    )
                    fg = pygame.Rect(x, y, dims["square_side"], dims["square_side"])
                    pygame.draw.rect(grid, (0, 0, 0), bg)
                    pygame.draw.rect(grid, color, fg)
        # highlight
        if self.highlight_position:
            idx, r, c = self.highlight_position
            shape = self.game_state.current_shapes[idx]
            if hasattr(shape, "form"):
                for i, row in enumerate(shape.form):
                    for j, val in enumerate(row):
                        if val:
                            x = (c + j) * (dims["square_side"] + self.grid_line_width)
                            y = (r + i) * (dims["square_side"] + self.grid_line_width)
                            highlight = pygame.Surface(
                                (dims["square_side"], dims["square_side"]),
                                pygame.SRCALPHA,
                            )
                            highlight.fill((255, 255, 0, 128))
                            grid.blit(highlight, (x, y))
        self.main_screen.blit(grid, (dims["grid_pos_x"], dims["grid_pos_y"]))

    def draw_shapes(self):
        dims = self.calculate_grid_dimensions()
        square = dims["grid_pos_x"] / 11.5
        cx = dims["grid_pos_x"] * 1.5 + dims["grid_side"] + 4
        my = self.main_screen.get_size()[1] // 4
        centers = [my, my * 2, my * 3]
        for idx in range(3):
            if idx < len(self.game_state.current_shapes):
                shape = self.game_state.current_shapes[idx]
                if hasattr(shape, "form"):
                    if idx == self.chosen_shape:
                        continue

                    blocks = []
                    for i, row in enumerate(shape.form):
                        for j, val in enumerate(row):
                            if val:
                                x = cx - (square * len(row) // 2) + j * (square + 2)
                                y = (
                                    centers[idx]
                                    - (square * len(shape.form) // 2)
                                    + i * (square + 2)
                                )
                                bg = pygame.Rect(x - 2, y - 2, square + 4, square + 4)
                                fg = pygame.Rect(x, y, square, square)
                                blocks.append(fg)

                                pygame.draw.rect(self.main_screen, (0, 0, 0), bg)
                                pygame.draw.rect(self.main_screen, shape.color, fg)
                                
                    if not hasattr(shape, "boundary"):
                        shape.boundary = pygame.Rect(
                            min(blocks, key=lambda r: r.left).left,
                            min(blocks, key=lambda r: r.top).top,
                            max(blocks, key=lambda r: r.right).right - min(blocks, key=lambda r: r.left).left,
                            max(blocks, key=lambda r: r.bottom).bottom - min(blocks, key=lambda r: r.top).top,
                        )

    def draw_cursor(self):
        """Draw the chosen shape at the cursor position for human play."""
        if (
            self.chosen_shape < 0
            or self.chosen_shape >= len(self.game_state.current_shapes)
            or not self.game_state.current_shapes[self.chosen_shape]
            or not hasattr(self.game_state.current_shapes[self.chosen_shape], "form")
        ):
            return

        dims = self.calculate_grid_dimensions()
        square_side = dims["square_side"]
        grid_pos_x = dims["grid_pos_x"]
        grid_pos_y = dims["grid_pos_y"]

        shape = self.game_state.current_shapes[self.chosen_shape]
        size = [len(shape.form), len(shape.form[0])]

        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Find closest grid cell to cursor
        if (
            grid_pos_x <= mouse_x <= grid_pos_x + dims["grid_side"]
            and grid_pos_y <= mouse_y <= grid_pos_y + dims["grid_side"]
        ):

            # Find the grid cell
            col = int((mouse_x - grid_pos_x) / (square_side + self.grid_line_width))
            row = int((mouse_y - grid_pos_y) / (square_side + self.grid_line_width))

            # Constrain to valid grid range
            col = max(0, min(7, col))
            row = max(0, min(7, row))

            # Adjust for shape size to show placement preview
            # Place shape so the center block would be at the selected cell
            col_offset = col - size[1] // 2
            row_offset = row - size[0] // 2

            # Make sure the placement is valid (constraining shapes at edges)
            if col_offset < 0:
                col_offset = 0
            if row_offset < 0:
                row_offset = 0
            if col_offset + size[1] > 8:
                col_offset = 8 - size[1]
            if row_offset + size[0] > 8:
                row_offset = 8 - size[0]

            # Draw shape preview at the grid-aligned position
            alpha_surface = pygame.Surface(
                (dims["grid_side"], dims["grid_side"]), pygame.SRCALPHA
            )

            for i in range(size[0]):
                for j in range(size[1]):
                    if shape.form[i][j]:
                        grid_row = row_offset + i
                        grid_col = col_offset + j

                        # Only draw if within grid bounds
                        if 0 <= grid_row < 8 and 0 <= grid_col < 8:
                            # Calculate position based on grid coordinates
                            pos_x = (square_side + self.grid_line_width) * grid_col
                            pos_y = (square_side + self.grid_line_width) * grid_row

                            # Draw semi-transparent shape preview
                            square = pygame.Rect(pos_x, pos_y, square_side, square_side)
                            color_with_alpha = (*shape.color[:3], 128)  # Add alpha
                            pygame.draw.rect(alpha_surface, color_with_alpha, square)

            # Store the current placement position for debugging
            self.current_placement = (row_offset, col_offset)

            # Blit the alpha surface to the main screen
            self.main_screen.blit(alpha_surface, (grid_pos_x, grid_pos_y))
        else:
            # Outside grid - follow cursor directly
            self.current_placement = None
            for i in range(size[0]):
                for j in range(size[1]):
                    if shape.form[i][j]:
                        pos_x = mouse_x - (square_side + self.grid_line_width) * (
                            size[1] // 2 - j
                        )
                        pos_y = mouse_y - (square_side + self.grid_line_width) * (
                            size[0] // 2 - i
                        )
                        square = pygame.Rect(pos_x, pos_y, square_side, square_side)
                        bg_square = pygame.Rect(
                            pos_x - 2, pos_y - 2, square_side + 4, square_side + 4
                        )
                        pygame.draw.rect(self.main_screen, (0, 0, 0), bg_square)
                        pygame.draw.rect(self.main_screen, shape.color, square)

    def draw_score(self):
        dims = self.calculate_grid_dimensions()
        pad = dims["grid_padding"]
        width = self.main_screen.get_size()[0]
        # High score
        fs = int(pad / 2)
        font_high = self.make_font(fs)
        self.main_screen.blit(
            font_high.render("HIGH SCORE", True, (135, 135, 135)), (pad // 3, pad // 6)
        )
        self.main_screen.blit(
            font_high.render(str(self.game_state.highest_score), True, (135, 135, 135)),
            (pad // 3, pad // 1.3),
        )
        # Current score
        fs = int(pad / 1.3)
        font_sc = self.make_font(fs)
        self.displayed_score = self.game_state.score
        text = font_sc.render(str(int(self.displayed_score)), True, (135, 135, 135))
        r = text.get_rect(center=(width // 2, pad // 2))
        self.main_screen.blit(text, r)

    def draw_combos(self):
        dims = self.calculate_grid_dimensions()
        pad = dims["grid_padding"]
        x = dims["grid_pos_x"]
        h = self.main_screen.get_size()[1]
        screen_w = x - pad // 5
        screen_h = x - pad // 2
        surf = pygame.Surface((screen_w, screen_h))
        surf.fill(self.BACKGROUND_COLOR)
        for i, msg in enumerate(reversed(self.game_state.combos[0])):
            font = self.make_font(int(pad / 3))
            surf.blit(
                font.render(msg, True, (135, 135, 135)),
                (0, screen_h - font.get_linesize() * (i + 1)),
            )
        self.main_screen.blit(surf, ((x / 2) - screen_w / 2, (h / 2) - screen_h / 2))

    def draw_game_over(self):
        w, h = self.main_screen.get_size()
        self.game_over_alpha = min(150, self.game_over_alpha + self.TRANSITION_SPEED)
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((220, 220, 220, self.game_over_alpha))
        self.main_screen.blit(overlay, (0, 0))
        # Text
        fs = w // 13
        font = self.make_font(fs)
        msg = (
            "NEW RECORD"
            if self.game_state.score > self.game_state.highest_score
            else "GAME OVER"
        )
        if msg == "NEW RECORD":
            self.game_state.highest_score = self.game_state.score
        text = font.render(msg, True, (80, 80, 80))
        self.main_screen.blit(text, text.get_rect(center=(w // 2, (h // 20) * 7.5)))
        # Score
        fs = w // 15
        font2 = self.make_font(fs)
        score = font2.render(str(self.game_state.score), True, (80, 80, 80))
        self.main_screen.blit(score, score.get_rect(center=(w // 2, (h // 20) * 12.5)))
        # Restart hint
        bg = pygame.Surface((w // 2, h // 15), pygame.SRCALPHA)
        bg.fill((255, 255, 255, 200))
        rect = bg.get_rect(center=(w // 2, (h // 20) * 16))
        self.main_screen.blit(bg, rect)
        hint = pygame.font.SysFont("Arial", w // 30).render(
            "Press SPACE to restart", True, (0, 0, 0)
        )
        self.main_screen.blit(hint, hint.get_rect(center=rect.center))

    def fade_in(self):
        """Fade in transition effect."""
        curr_main_width, curr_main_height = self.main_screen.get_size()

        self.fade_alpha = max(0, self.fade_alpha - self.TRANSITION_SPEED)

        transition_screen = pygame.Surface(
            (curr_main_width, curr_main_height), pygame.SRCALPHA
        )
        transition_screen.fill((220, 220, 220, self.fade_alpha))
        self.main_screen.blit(transition_screen, (0, 0))

        if self.fade_alpha == 0:
            self.transition_in = False

    def fade_out(self):
        """Fade out transition effect."""
        curr_main_width, curr_main_height = self.main_screen.get_size()

        self.fade_alpha = min(255, self.fade_alpha + self.TRANSITION_SPEED)

        transition_screen = pygame.Surface(
            (curr_main_width, curr_main_height), pygame.SRCALPHA
        )
        transition_screen.fill((220, 220, 220, self.fade_alpha))
        self.main_screen.blit(transition_screen, (0, 0))

        if self.fade_alpha == 255:
            self.transition_out = False
            # Reset can be triggered here if needed

    def draw_agent_thinking(self):
        w, _ = self.main_screen.get_size()
        surf = pygame.Surface((w, 100), pygame.SRCALPHA)
        surf.fill((220, 220, 220, 200))
        fs = 36
        font = self.make_font(fs)
        surf.blit(font.render("Agent thinking...", True, (0, 0, 0)), (20, 20))
        if self.highlight_position:
            idx, r, c = self.highlight_position
            act = f"Placing Shape {idx+1} at ({r},{c})"
            surf.blit(font.render(act, True, (0, 0, 0)), (20, 60))
        self.main_screen.blit(surf, (0, 0))

    def set_agent_action(self, shape_idx, row, col):
        """Set the position for highlighting the agent's planned move."""
        self.highlight_position = (shape_idx, row, col)

    def set_agent_thinking(self, is_thinking):
        """Toggle the agent thinking visualization."""
        self.agent_thinking = is_thinking

    def process_human_events(self):
        """Process human input events and return actions if taken.

        Returns:
            "RESET" if reset requested
            (shape_idx, row, col) if action taken
            None if no action taken
        """
        action_taken = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"  # tell the caller we want to quit

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # End the game when ESC is pressed
                    self.game_state.game_over = True
                    print("Game ended by player.")
                    return "RESET"  # Indicate reset action

                # Toggle debug mode
                if event.key == pygame.K_F3:
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'enabled' if self.debug_mode else 'disabled'}")

                # Restart on space if game over
                if event.key == pygame.K_SPACE and self.game_state.game_over:
                    self.transition_in = True
                    self.fade_alpha = 255
                    self.game_over_alpha = 0
                    return "RESET"

                # Space to place shape (alternative to mouse click)
                if (
                    event.key == pygame.K_SPACE
                    and self.chosen_shape != -1
                    and not self.game_state.game_over
                ):
                    if self.current_placement is not None:
                        row, col = self.current_placement
                        action_taken = (self.chosen_shape, row, col)

            # Handle mouse click for placement
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and not self.game_state.game_over
            ):
                for idx, shape in enumerate(self.game_state.current_shapes):
                    if (
                        idx != self.chosen_shape
                        and hasattr(shape, "form")
                        and shape.boundary.collidepoint(event.pos)
                    ):
                        self.chosen_shape = idx
                        return None

                if (
                    self.chosen_shape != -1
                    and self.current_placement is not None
                ):
                    row, col = self.current_placement
                    action_taken = (self.chosen_shape, row, col)

        return action_taken

    def get_rgb_array(self):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.main_screen)), axes=(1, 0, 2)
        )

    def close(self):
        pygame.quit()
