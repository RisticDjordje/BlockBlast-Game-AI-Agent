import pygame
import sys
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
        self.INIT_HEIGHT = 1000
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

        # Load sounds and assets
        try:
            self.clear_sound = pygame.mixer.Sound(r"Assets/clear_sf.wav")
        except:
            # Create a dummy sound if file not found
            self.clear_sound = pygame.mixer.Sound(buffer=bytearray(10))
            print("Warning: Clear sound file not found, using silent sound.")

        # Try to load the game font
        try:
            self.font = pygame.font.Font(r"Assets/LECO.ttf", 24)
        except:
            self.font = pygame.font.SysFont("Arial", 24)
            print("Warning: Font file not found, using system font.")

        # Setup visualization variables
        self.chosen_shape = -1  # For human play mode
        self.agent_thinking = False
        self.highlight_position = None
        self.displayed_score = 0
        self.game_over_alpha = 0
        self.current_placement = None  # For tracking placement position

        # Fade transition variables
        self.fade_alpha = 0
        self.transition_in = False
        self.transition_out = False
        self.TRANSITION_SPEED = 15

        # Add debug visualization
        self.debug_mode = False

    def render(self):
        """Render the current state of the game."""
        # Fill background
        self.main_screen.fill(self.BACKGROUND_COLOR)

        # Draw main elements
        self.draw_grid()
        self.draw_shapes()
        self.draw_score()
        self.draw_combos()

        # Draw cursor for human play
        if self.chosen_shape != -1:
            self.draw_cursor()

        # Draw agent thinking visualization if enabled
        if self.agent_thinking:
            self.draw_agent_thinking()

        # Draw game over overlay if the game is over
        if self.game_state.game_over:
            self.draw_game_over()

        # Handle transition effects
        if self.transition_in:
            self.fade_in()
        elif self.transition_out:
            self.fade_out()

        # Draw debug info if enabled
        if self.debug_mode and self.current_placement:
            self.draw_debug_info()

        # Update display
        pygame.display.flip()

        # Control frame rate
        self.clock.tick(self.FPS)

    def draw_debug_info(self):
        """Draw debug information on screen."""
        debug_font = pygame.font.SysFont("Arial", 16)
        y_pos = 10

        # Draw current placement position
        if self.current_placement:
            row, col = self.current_placement
            debug_text = f"Placement position: ({row}, {col})"
            text_surface = debug_font.render(debug_text, True, (255, 0, 0))
            self.main_screen.blit(text_surface, (10, y_pos))
            y_pos += 20

        # Draw current mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        debug_text = f"Mouse position: ({mouse_x}, {mouse_y})"
        text_surface = debug_font.render(debug_text, True, (255, 0, 0))
        self.main_screen.blit(text_surface, (10, y_pos))
        y_pos += 20

        # Draw grid dimensions
        dims = self.calculate_grid_dimensions()
        debug_text = f"Grid pos: ({int(dims['grid_pos_x'])}, {int(dims['grid_pos_y'])}), Cell size: {dims['square_side']:.1f}"
        text_surface = debug_font.render(debug_text, True, (255, 0, 0))
        self.main_screen.blit(text_surface, (10, y_pos))

    def calculate_grid_dimensions(self):
        """Calculate the grid dimensions based on the current window size."""
        curr_main_width, curr_main_height = self.main_screen.get_size()
        grid_padding = curr_main_height // 10  # Top and bottom padding

        # Calculate grid size (make sure it's divisible by 8)
        grid_side = curr_main_height - 2 * grid_padding
        grid_side = grid_side - (grid_side % 8) + (self.grid_line_width * 7)

        grid_pos_x = (curr_main_width / 2) - (grid_side / 2)
        grid_pos_y = grid_padding

        # Size of each cell in the grid
        square_side = (grid_side - self.grid_line_width * 7) / 8

        return {
            "grid_padding": grid_padding,
            "grid_side": grid_side,
            "grid_pos_x": grid_pos_x,
            "grid_pos_y": grid_pos_y,
            "square_side": square_side,
        }

    def draw_grid(self):
        """Draw the game grid with the current state."""
        dims = self.calculate_grid_dimensions()
        grid_side = dims["grid_side"]
        grid_pos_x = dims["grid_pos_x"]
        grid_pos_y = dims["grid_pos_y"]
        square_side = dims["square_side"]

        # Create grid surface
        grid_screen = pygame.Surface((grid_side, grid_side))
        grid_screen.fill((255, 255, 255))

        # Draw outline rectangle
        rect_padding = 2
        outline_rect_side = grid_side + 2 * rect_padding
        outline_rect = pygame.Rect(
            grid_pos_x - rect_padding,
            grid_pos_y - rect_padding,
            outline_rect_side,
            outline_rect_side,
        )
        pygame.draw.rect(self.main_screen, (0, 0, 0), outline_rect)

        # Draw grid lines
        start_lines_pos = (grid_side - self.grid_line_width * 7) / 8
        for i in range(7):
            pygame.draw.line(
                grid_screen,
                (200, 200, 200),
                (0, start_lines_pos),
                (grid_side, start_lines_pos),
                self.grid_line_width,
            )
            pygame.draw.line(
                grid_screen,
                (200, 200, 200),
                (start_lines_pos, 0),
                (start_lines_pos, grid_side),
                self.grid_line_width,
            )
            start_lines_pos += (
                grid_side - self.grid_line_width * 7
            ) / 8 + self.grid_line_width

        # Draw filled squares
        for i in range(8):
            for j in range(8):
                if self.game_state.grid[i][j]:
                    pos_x = (square_side + self.grid_line_width) * j
                    pos_y = (square_side + self.grid_line_width) * i
                    square = pygame.Rect(pos_x, pos_y, square_side, square_side)
                    bg_square = pygame.Rect(
                        pos_x - 2, pos_y - 2, square_side + 4, square_side + 4
                    )
                    pygame.draw.rect(grid_screen, (0, 0, 0), bg_square)
                    pygame.draw.rect(grid_screen, self.game_state.grid[i][j], square)

        # Draw highlight for agent's planned move if applicable
        if self.highlight_position:
            shape_idx, row, col = self.highlight_position
            if (
                shape_idx >= 0
                and shape_idx < len(self.game_state.current_shapes)
                and self.game_state.current_shapes[shape_idx]
            ):

                shape = self.game_state.current_shapes[shape_idx]
                if hasattr(shape, "form"):
                    size = [len(shape.form), len(shape.form[0])]

                    for i in range(size[0]):
                        for j in range(size[1]):
                            if shape.form[i][j]:
                                pos_x = (square_side + self.grid_line_width) * (col + j)
                                pos_y = (square_side + self.grid_line_width) * (row + i)
                                square = pygame.Rect(
                                    pos_x, pos_y, square_side, square_side
                                )
                                # Draw semi-transparent highlight
                                highlight = pygame.Surface(
                                    (square_side, square_side), pygame.SRCALPHA
                                )
                                highlight.fill(
                                    (255, 255, 0, 128)
                                )  # Semi-transparent yellow
                                grid_screen.blit(highlight, square)

        # Blit grid to main screen
        self.main_screen.blit(grid_screen, (grid_pos_x, grid_pos_y))

    def draw_shapes(self):
        """Draw the available shapes on the side of the screen."""
        dims = self.calculate_grid_dimensions()
        grid_pos_x = dims["grid_pos_x"]
        grid_pos_y = dims["grid_pos_y"]
        grid_side = dims["grid_side"]
        curr_main_width, curr_main_height = self.main_screen.get_size()

        # Size of each square in the shapes
        square_side = grid_pos_x // 11.5

        # Position for the shapes on the right side
        shapes_margin = curr_main_width // 4
        center_x = [shapes_margin, shapes_margin * 2, shapes_margin * 3]
        center_y = (grid_pos_y * 1.5) + (grid_side + 4)

        # Draw each available shape
        for cshape in range(3):
            # Check if shape index is valid
            if cshape < len(self.game_state.current_shapes):
                shape = self.game_state.current_shapes[cshape]
                # Check if shape is valid
                if shape and hasattr(shape, "form"):
                    size = [len(shape.form), len(shape.form[0])]

                    # Skip if this is the chosen shape for human play
                    if cshape == self.chosen_shape:
                        continue

                    # Draw each block of the shape
                    bounds = []
                    for i in range(size[0]):
                        for j in range(size[1]):
                            if shape.form[i][j]:
                                pos_x = (
                                    center_x[cshape]
                                    - (square_side * size[1] // 2)
                                    + j * (square_side + 2)
                                )
                                pos_y = (
                                    center_y
                                    - (square_side * size[0] // 2)
                                    + i * (square_side + 2)
                                )

                                # Draw the square
                                square = pygame.Rect(
                                    pos_x, pos_y, square_side, square_side
                                )
                                bg_square = pygame.Rect(
                                    pos_x - 2,
                                    pos_y - 2,
                                    square_side + 4,
                                    square_side + 4,
                                )

                                pygame.draw.rect(self.main_screen, (0, 0, 0), bg_square)
                                pygame.draw.rect(self.main_screen, shape.color, square)

                                bounds.append(square)

                    shape.bound = pygame.Rect(
                        min([b.left for b in bounds]),
                        min([b.top for b in bounds]),
                        max([b.right for b in bounds]) - min([b.left for b in bounds]),
                        max([b.bottom for b in bounds]) - min([b.top for b in bounds]),
                    )

                    # No longer draw the hint 
                    if False:
                        # Draw key hint
                        if cshape == 0:
                            key_text = "E"
                        elif cshape == 1:
                            key_text = "R"
                        else:
                            key_text = "T"

                        text = self.font.render(key_text, True, (0, 0, 0))
                        text_rect = text.get_rect(
                            center=(center_x, center_y[cshape] - square_side * 1.5)
                        )
                        self.main_screen.blit(text, text_rect)

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
        """Draw the current score and high score."""
        dims = self.calculate_grid_dimensions()
        grid_padding = dims["grid_padding"]
        curr_main_width = self.main_screen.get_size()[0]

        # Draw high score
        font_size = int(grid_padding / 2)
        font_highest = self.font
        try:
            font_highest = pygame.font.Font(r"Assets/LECO.ttf", font_size)
        except:
            font_highest = pygame.font.SysFont("Arial", font_size)

        text_highest = font_highest.render("HIGH SCORE", True, (255, 215, 0))
        text_highest_value = font_highest.render(
            str(self.game_state.highest_score), True, (255, 215, 0)
        )
        self.main_screen.blit(text_highest, (grid_padding // 3, grid_padding // 6))
        self.main_screen.blit(
            text_highest_value, (grid_padding // 3, grid_padding // 1.3)
        )

        # Draw current score with smooth animation
        font_size = int(grid_padding / 1.3)
        try:
            font_score = pygame.font.Font(r"Assets/LECO.ttf", font_size)
        except:
            font_score = pygame.font.SysFont("Arial", font_size)

        self.displayed_score = self.game_state.score

        score_text = font_score.render(
            str(int(self.displayed_score)), True, (255, 215, 0)
        )
        score_rect = score_text.get_rect()
        score_rect.center = (curr_main_width // 2, grid_padding // 2)
        self.main_screen.blit(score_text, score_rect)

    def draw_combos(self):
        """Draw the combo information."""
        dims = self.calculate_grid_dimensions()
        grid_padding = dims["grid_padding"]
        grid_pos_x = dims["grid_pos_x"]
        curr_main_width, curr_main_height = self.main_screen.get_size()

        # Setup combo display area
        combos_padding = grid_pos_x // 10
        combo_screen_x = grid_pos_x - combos_padding * 2
        combo_screen_y = grid_pos_x - combos_padding * 4

        # Create combo display surface
        combos_screen = pygame.Surface((combo_screen_x, combo_screen_y))
        combos_screen.fill(self.BACKGROUND_COLOR)

        # Position the combo display
        combo_pos_x = curr_main_width - grid_pos_x + grid_padding / 4 # (grid_pos_x / 2) - (combo_screen_x / 2)
        combo_pos_y = (curr_main_height / 2) - (combo_screen_y / 2)

        # Draw each combo line
        j = 1
        for i in range(len(self.game_state.combos[0]) - 1, -1, -1):
            font_size = int(grid_padding / 3)
            font_combo = self.font
            try:
                font_combo = pygame.font.Font(r"Assets/LECO.ttf", font_size)
            except:
                font_combo = pygame.font.SysFont("Arial", font_size)

            text_combo = font_combo.render(
                self.game_state.combos[0][i], True, (255, 215, 0)
            )
            combos_screen.blit(text_combo, (0, combo_screen_y - font_size * j))
            j += 1

        # Blit combo screen to main display
        self.main_screen.blit(combos_screen, (combo_pos_x, combo_pos_y))

    def draw_game_over(self):
        """Draw game over overlay."""
        curr_main_width, curr_main_height = self.main_screen.get_size()

        # Animate alpha, but make it less dim (max 150 instead of 220)
        self.game_over_alpha = min(150, self.game_over_alpha + self.TRANSITION_SPEED)

        # Create semi-transparent overlay
        transition_screen = pygame.Surface(
            (curr_main_width, curr_main_height), pygame.SRCALPHA
        )
        transition_screen.fill((220, 220, 220))
        transition_screen.set_alpha(self.game_over_alpha)

        # Create text overlay
        text_screen = pygame.Surface(
            (curr_main_width, curr_main_height), pygame.SRCALPHA
        )
        text_screen.set_alpha(255)  # Full opacity for text

        # Game over text
        game_over_text = "GAME OVER"
        if self.game_state.score > self.game_state.highest_score:
            game_over_text = "NEW RECORD"
            self.game_state.highest_score = self.game_state.score

        # Render text
        font_size = curr_main_width // 13
        try:
            font_text = pygame.font.Font(r"Assets/LECO.ttf", font_size)
        except:
            font_text = pygame.font.SysFont("Arial", font_size)

        text = font_text.render(game_over_text, True, (80, 80, 80))  # Darker text
        text_rect = text.get_rect()
        text_rect.center = (curr_main_width // 2, (curr_main_height // 20) * 7.5)
        text_screen.blit(text, text_rect)

        # Render score
        font_size = curr_main_width // 15
        try:
            font_score = pygame.font.Font(r"Assets/LECO.ttf", font_size)
        except:
            font_score = pygame.font.SysFont("Arial", font_size)

        score_text = font_score.render(
            str(self.game_state.score), True, (80, 80, 80)
        )  # Darker text
        score_rect = score_text.get_rect()
        score_rect.center = (curr_main_width // 2, (curr_main_height // 20) * 12.5)
        text_screen.blit(score_text, score_rect)

        # Add a background for better readability
        restart_bg = pygame.Surface(
            (curr_main_width // 2, curr_main_height // 15), pygame.SRCALPHA
        )
        restart_bg.fill((255, 255, 255, 200))  # Semi-transparent white
        restart_bg_rect = restart_bg.get_rect()
        restart_bg_rect.center = (curr_main_width // 2, (curr_main_height // 20) * 16)
        text_screen.blit(restart_bg, restart_bg_rect)

        # Render instructions to restart
        font_size = curr_main_width // 30
        font_restart = pygame.font.SysFont("Arial", font_size)
        restart_text = font_restart.render("Press SPACE to restart", True, (0, 0, 0))
        restart_rect = restart_text.get_rect()
        restart_rect.center = (curr_main_width // 2, (curr_main_height // 20) * 16)
        text_screen.blit(restart_text, restart_rect)

        # Blit overlays
        self.main_screen.blit(transition_screen, (0, 0))
        self.main_screen.blit(text_screen, (0, 0))

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
        """Visualize agent's decision-making process."""
        curr_main_width, curr_main_height = self.main_screen.get_size()

        # Create a semi-transparent overlay for the top
        thinking_surface = pygame.Surface((curr_main_width, 100), pygame.SRCALPHA)
        thinking_surface.fill((220, 220, 220, 200))

        # Draw "Agent thinking..." text
        font_size = 36
        try:
            font = pygame.font.Font(r"Assets/LECO.ttf", font_size)
        except:
            font = pygame.font.SysFont("Arial", font_size)

        text = font.render("Agent thinking...", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topleft = (20, 20)
        thinking_surface.blit(text, text_rect)

        # If we have a highlighted position, show what action it's considering
        if self.highlight_position:
            shape_idx, row, col = self.highlight_position
            action_text = f"Placing Shape {shape_idx+1} at position ({row}, {col})"
            action_surf = font.render(action_text, True, (0, 0, 0))
            action_rect = action_surf.get_rect()
            action_rect.topleft = (20, 60)
            thinking_surface.blit(action_surf, action_rect)

        # Blit the thinking surface
        self.main_screen.blit(thinking_surface, (0, 0))

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

            print(f"Event: {event}")
            
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

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

                # Select shapes using E, R, T keys
                if (
                    event.key in [pygame.K_e, pygame.K_r, pygame.K_t]
                    and not self.game_state.game_over
                ):
                    mapping = {pygame.K_e: 0, pygame.K_r: 1, pygame.K_t: 2}
                    shape_idx = mapping[event.key]

                    # Toggle selection
                    if shape_idx == self.chosen_shape:
                        self.chosen_shape = -1
                    else:
                        # Check if shape is valid
                        if (
                            shape_idx < len(self.game_state.current_shapes)
                            and self.game_state.current_shapes[shape_idx]
                            and hasattr(
                                self.game_state.current_shapes[shape_idx], "form"
                            )
                        ):
                            self.chosen_shape = shape_idx

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
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                for i in range(3):
                    shape = self.game_state.current_shapes[i]
                    if shape and hasattr(shape, "form") and hasattr(shape, "bound"):
                        # print(f"Shape {i} bounds: {shape.bound}, {mouse_x}, {mouse_y}")
                        if (
                            shape.bound.collidepoint(mouse_x, mouse_y)
                            and i != self.chosen_shape
                        ):
                            self.chosen_shape = i
                 
                if self.chosen_shape != -1 and self.current_placement is not None:
                    row, col = self.current_placement
                    action_taken = (self.chosen_shape, row, col)

        return action_taken

    def _get_action_from_mouse(self, mouse_x, mouse_y):
        """Convert mouse position to grid action."""
        dims = self.calculate_grid_dimensions()
        grid_pos_x = dims["grid_pos_x"]
        grid_pos_y = dims["grid_pos_y"]
        grid_side = dims["grid_side"]
        square_side = dims["square_side"]

        # Check if click is within grid
        if (
            grid_pos_x <= mouse_x <= grid_pos_x + grid_side
            and grid_pos_y <= mouse_y <= grid_pos_y + grid_side
        ):

            # Calculate grid coordinates using the same logic as in draw_cursor
            col = int((mouse_x - grid_pos_x) / (square_side + self.grid_line_width))
            row = int((mouse_y - grid_pos_y) / (square_side + self.grid_line_width))

            # Constrain to valid grid range
            col = max(0, min(7, col))
            row = max(0, min(7, row))

            # Get the shape dimensions
            shape = self.game_state.current_shapes[self.chosen_shape]
            if not shape or not hasattr(shape, "form"):
                return None

            size = [len(shape.form), len(shape.form[0])]

            # Calculate offsets exactly as in draw_cursor to ensure consistency
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

            # Return the top-left corner for placement
            return (self.chosen_shape, row_offset, col_offset)

        return None

    def get_rgb_array(self):
        """Convert the pygame surface to a numpy array for the 'rgb_array' render mode."""
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.main_screen)), axes=(1, 0, 2)
        )

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
