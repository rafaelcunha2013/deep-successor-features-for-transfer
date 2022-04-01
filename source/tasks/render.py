"""
 Pygame base template for opening a window

 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/

 Explanation video: http://youtu.be/vRB_983kUMc
"""

import pygame


class Render:
    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)

    def __init__(self, maze):
        # Environment that will be showed
        self.maze_height, self.maze_width = maze.shape
        self.maze = maze

        # Square dimensions
        self.WIDTH = 20
        self.HEIGHT = 20
        self.MARGIN = 1
        self.number_of_squares = self.maze_height



        # Set the width and height of the screen [width, height]
        grid_lenth = self.number_of_squares * (self.WIDTH + self.MARGIN)
        self.size = (grid_lenth, grid_lenth)
        self.screen = pygame.display.set_mode(self.size)
        self.initialize()

    def update(self, state):
        if isinstance(state[0], tuple):
            for single_state in self.agent_state:
                r, c = single_state
                color = Render.YELLOW
                self.draw_shape(r, c, color, 'rect')
            for single_state in state:
                r, c = single_state
                color = Render.BLUE
                self.draw_shape(r, c, color, 'circle')
            self.agent_state = state
        else:
            r, c = self.agent_state[0]
            color = Render.YELLOW
            self.draw_shape(r, c, color, 'rect')
            r, c = state
            color = Render.BLUE
            self.draw_shape(r, c, color, 'circle')

            self.agent_state = [state]


        pygame.display.flip()
        pygame.time.delay(200)

    def initialize(self):
        # Agent state
        self.agent_state = []

        pygame.init()
        # self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("GridWorld")

        # Loop until the user clicks the close button.
        #done = False

        # Used to manage how fast the screen updates
        clock = pygame.time.Clock()
        # -------- Main Program Loop -----------
        #while not done:
        # --- Main event loop
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         done = True

        # --- Game logic should go here

        # --- Screen-clearing code goes here

        # Here, we clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.

        # If you want a background image, replace this clear with blit'ing the
        # background image.
        self.screen.fill(Render.BLACK)

        # --- Drawing code should go here
        # color = Render.GREEN
        for r in range(self.maze_height):
            for c in range(self.maze_width):
                color = Render.WHITE
                self.draw_shape(r, c, color, 'rect')

        for c in range(self.maze_width):
            for r in range(self.maze_height):
                if self.maze[r, c] == 'G':
                    color = Render.GREEN
                    self.draw_shape(r, c, color, 'circle')
                elif self.maze[r, c] == ' ':
                    color = Render.WHITE
                    self.draw_shape(r, c, color, 'rect')
                elif self.maze[r, c] == '_':
                    self.agent_state.append((r, c))
                    color = Render.BLUE
                    self.draw_shape(r, c, color, 'circle')
                elif self.maze[r, c] == 'X':
                    color = Render.BLACK
                    self.draw_shape(r, c, color, 'rect')
                elif self.maze[r, c] in {'3'}:
                    color = Render.GREEN
                    self.draw_shape(r, c, color, 'tri')
                elif self.maze[r, c] in {'2'}:
                    color = Render.RED
                    self.draw_shape(r, c, color, 'tri')
                elif self.maze[r, c] in {'1'}:
                    color = Render.BLUE
                    self.draw_shape(r, c, color, 'tri')
                # pygame.draw.rect(self.screen,
                #                  color,
                #                  [(self.MARGIN + self.WIDTH) * c + self.MARGIN,
                #                   (self.MARGIN + self.HEIGHT) * r + self.MARGIN,
                #                   self.WIDTH,
                #                   self.HEIGHT])

        # --- Go ahead and update the screen with what we've drawn.

        pygame.display.flip()

        # --- Limit to 60 frames per second
        clock.tick(60)

        # Close the window and quit.
        #pygame.quit()

    def draw_shape(self, r, c, color, shape):
        if shape == 'rect':
            pygame.draw.rect(self.screen,
                             color,
                             [(self.MARGIN + self.WIDTH) * c + self.MARGIN,
                              (self.MARGIN + self.HEIGHT) * r + self.MARGIN,
                              self.WIDTH,
                              self.HEIGHT])
        elif shape == 'circle':
            pygame.draw.circle(self.screen,
                               color,
                               [(self.MARGIN + self.WIDTH) * c + self.MARGIN + self.WIDTH/2,
                                (self.MARGIN + self.HEIGHT) * r + self.MARGIN + self.HEIGHT/2],
                               10)
        elif shape == 'tri':
            pygame.draw.polygon(self.screen,
                               color,
                               [[(self.MARGIN + self.WIDTH) * c + self.MARGIN + self.WIDTH/2,
                                (self.MARGIN + self.HEIGHT) * r + self.MARGIN],
                               [(self.MARGIN + self.WIDTH) * c + self.MARGIN,
                                (self.MARGIN + self.HEIGHT) * r + self.MARGIN + self.HEIGHT],
                               [(self.MARGIN + self.WIDTH) * c + self.MARGIN + self.WIDTH,
                                (self.MARGIN + self.HEIGHT) * r + self.MARGIN + self.HEIGHT]])


if __name__ == "__main__":
    my_grid = Render()
    my_grid.update()