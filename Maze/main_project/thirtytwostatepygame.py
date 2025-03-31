import pygame
import random
import numpy as np
from modules.mapping import STATES_TO_COORDINATES, STATES_TO_ORIENTATIONS, LEGAL_MOVEMENT

pygame.init()

#further refactoring (organize file structure)
class Environment:
    #agent, #reward, #maze
    def __init__(self, maze_width, maze_height, goal_corner = 0, start = 0):
        
        #maze defining
        self.maze = [[0] * maze_width for _ in range(maze_height)]
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.goal_corner = goal_corner

        #write maze
        self.write_open_tiles()
        self.write_goals()

        #trial defining

        #agent defining
        self.start = random.choice([0, 1, 2, 3])
        self.configure_doors_for_goal_seeking()
        cell = self.identify_start_cell(self.start)
        self.y = cell[1]
        self.x = cell[0]
        print("x = ", self.x, "y =", self.y)
        self.state = self.identify_state(self.x, self.y)
        self._infer_orientation()
        print(self.start)

    def _infer_orientation(self): # left = 0, right = 1
        self.orientation = STATES_TO_ORIENTATIONS[self.state-1]

    def identify_state(self, x, y):
        state = STATES_TO_COORDINATES.index((x, y)) + 1 if ((x, y) in STATES_TO_COORDINATES) else -1
        return state

    def write_open_tiles(self): # maze_width and maze_height
        for x in range(1,self.maze_width-1):
            self.maze[1][x] = 1
            self.maze[self.maze_height-2][x] = 1
            self.maze[self.maze_height//2][x] = 1
        for y in range(1,self.maze_height-1):
            self.maze[y][self.maze_width-2] = 1
            self.maze[y][self.maze_width//2] = 1
            self.maze[y][1] = 1
    
    def write_goals(self): #goal_corner
        self.maze[0][0] = 2 if self.goal_corner == 0 else 3
        self.maze[0][self.maze_width-1] = 2 if self.goal_corner == 1 else 3
        self.maze[self.maze_height-1][0] = 2 if self.goal_corner == 3 else 3
        self.maze[self.maze_height-1][self.maze_width-1] = 2 if self.goal_corner == 2 else 3

    def configure_doors_for_goal_seeking(self):
        self.potential_goal_seeking_doors = [1,9,11,3,13,15,5,17,19,7,21,23]
        self.potential_return_doors = [2,9,11,4,13,15, 8, 21, 23, 6, 17, 19]
        #3, 13, 15, 9, 11 for start 0
        #13, 15, 1, 9, 11 for start 2
        primary_offset = 0
        if self.start == 0:
            offset = 1
        elif self.start == 2:
            offset = 4
        if self.start == 1:
            offset = 4
            primary_offset = 6
        if self.start == 3:
            offset = 1
            primary_offset = 6
        for i in range(0,5):
            self.maze[STATES_TO_COORDINATES[self.potential_goal_seeking_doors[primary_offset + (offset+i)%6]-1][1]][STATES_TO_COORDINATES[self.potential_goal_seeking_doors[primary_offset + (offset+i)%6]-1][0]] = 4

    def identify_start_cell(self, start):
        if start == 0:
            return (self.maze_height//2, self.maze_width//2 - 1)
        elif start == 1:
            return (self.maze_height//2 + 1, self.maze_width//2)
        elif start == 2:
            return (self.maze_height//2, self.maze_width//2 + 1)
        elif start == 3:
            return (self.maze_height//2 - 1, self.maze_width//2)


class Screen:
    def __init__(self, width = 810, height = 860, cell_size = 90):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        #colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_RED = (255, 100, 100)
        self.BLUE = (0,0,255)

        #pygame
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Maze Game")


    def draw_maze(self, maze, height, width):
        for y in range(height):
            for x in range(width):
                if maze[y][x] == 1: # empty space
                    pygame.draw.rect(self.screen, self.GRAY, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
                elif maze[y][x] == 0: # wall
                    pygame.draw.rect(self.screen, self.BLACK, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
                elif maze[y][x] == 2: # goal
                    pygame.draw.rect(self.screen, self.GREEN, (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))
                elif maze[y][x] == 3: # empty well
                    pygame.draw.rect(self.screen, self.WHITE, (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))
                elif maze[y][x] == 4: # door
                    pygame.draw.rect(self.screen, self.BLUE, (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))
        
    def draw_agent(self, x, y, orientation):
        direction_points = [
        (x * self.cell_size + self.cell_size, y * self.cell_size + self.cell_size),  # bottom right
        (x * self.cell_size, y * self.cell_size + self.cell_size),  # bottom left
        (x * self.cell_size, y * self.cell_size),  # top left
        (x * self.cell_size + self.cell_size, y * self.cell_size),  # top right
        (x * self.cell_size + (self.cell_size // 2), y * self.cell_size),  # top middle
        (x * self.cell_size + self.cell_size, y * self.cell_size + (self.cell_size // 2)),  # right middle
        (x * self.cell_size + (self.cell_size // 2), y * self.cell_size + self.cell_size),  # bottom middle
        (x * self.cell_size, y * self.cell_size + (self.cell_size // 2)),  # left middle
        ]
        
        font = pygame.font.SysFont(None, 30)
    
        text = font.render(f"Location: ({x}, {y})   State: currently undefined", True, self.WHITE)
        self.screen.blit(text, (300, 10))
        pygame.draw.polygon(self.screen, self.RED, [direction_points[orientation % 4], direction_points[(orientation+1)%4],direction_points[4+(orientation)%4]])

    def draw(self, env):
        maze = env.maze
        self.draw_maze(maze, env.maze_width, env.maze_height)
        self.draw_agent(env.x, env.y, env.orientation)

        
def main():
    screen = Screen()
    maze_width = screen.width//screen.cell_size
    maze_height = (screen.height-50)//screen.cell_size
    environment = Environment(maze_width, maze_height)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                print(model.choose_action(STATES_TO_COORDINATES.index((agent.y, agent.x))))
            if event.key == pygame.K_UP:
                agent.move(0, maze)
            elif event.key == pygame.K_DOWN:
                agent.move(3, maze)
            elif event.key == pygame.K_LEFT:
                agent.move(1, maze)
            elif event.key == pygame.K_RIGHT:
                agent.move(2, maze)
        screen.draw(environment)
        pygame.display.flip()


if __name__ == "__main__":
    main()