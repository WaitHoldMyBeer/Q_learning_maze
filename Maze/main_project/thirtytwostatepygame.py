import pygame
import random
import numpy as np
from modules.mapping import STATES_TO_COORDINATES, STATES_TO_ORIENTATIONS, LEGAL_MOVEMENT

pygame.init()

class QLearningModel:
    def __init__(self, n_states = 32, n_actions = 4, alpha = 0.9, # learning rate
    gamma = 0.95, # discount factor
    epsilon = 1.0, # exploration rate
    epsilon_decay = 0.8, # decay rate for epsilon
    min_epsilon = 0.01, # minimum exploration rate
    block_size = 32, # number of episodes to train
    max_steps = 200 # maximum steps per episode
    ):
        self.action_space = [0,1,2,3] # forward, left, right, about face
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        self.learning_rate = alpha
        self.exploration_probability = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.block_size = block_size
        self.max_steps = max_steps
        self.discount_factor = gamma
        self.cumulative_reward = 0
    def choose_action(self, state):
        if random.uniform(0,1) < self.exploration_probability:
            return random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])
    def step(self, env):
        
        action = self.choose_action(env.state-1)
        print(action)
        print("exploration rate =", self.exploration_probability)
        old_value = self.q_table[env.state-1, action]
        reward, new_state = env.location_move(action)

        next_max = np.max(self.q_table[new_state-1, :])
        self.q_table[env.state-1, action] = (1-self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.exploration_probability = max(self.min_epsilon, self.exploration_probability * self.epsilon_decay)
        self.cumulative_reward = self.cumulative_reward + reward
        print("cumulative_reward = ",self.cumulative_reward)
        print(self.q_table[env.state-1, :])

#further refactoring (organize file structure)
class Environment:
    #agent, #reward, #maze
    def __init__(self, maze_width, maze_height, goal_corner = 0, start = [0,1,2,3]):

        #maze defining
        self.maze = [[0] * maze_width for _ in range(maze_height)]
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.goal_corner = goal_corner
        self.goal_corner_state = self.identify_state(self.find_goal_corners(self.goal_corner)[0], self.find_goal_corners(self.goal_corner)[1])

        #write maze
        self.write_open_tiles()
        self.write_goals()

        #trial defining

        #agent defining
        self.start_options = start
        self.start = random.choice(self.start_options)
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

    def find_goal_corners(self, goal_corner):
        if goal_corner == 0:
            return (0,0)
        elif goal_corner == 1:
            return (0,self.maze_width-1)
        elif goal_corner == 2:
            return (self.maze_height-1, self.maze_width-1)
        elif goal_corner == 3:
            return (self.maze_height-1, 0)

    def write_goals(self): #goal_corner
        goal_corners = self.find_goal_corners(self.goal_corner)
        self.maze[0][0] = 3
        self.maze[0][self.maze_width-1] = 3
        self.maze[self.maze_height-1][0] = 3
        self.maze[self.maze_height-1][self.maze_width-1] = 3
        self.maze[goal_corners[0]][goal_corners[1]] = 2 #goal

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
    
    def location_move(self, move):
        moves = [0,1,2,3]
        self.state = self.identify_state(self.x, self.y)

        # check if new state is valid
        new_state = LEGAL_MOVEMENT[self.state-1][moves[move]] if LEGAL_MOVEMENT[self.state-1][moves[move]] != 0 else self.state

        # checks if there is a door within the rectangle formed with the corners of new and current state
        contains_door = False
        for i in range(min(STATES_TO_COORDINATES[self.state-1][0], STATES_TO_COORDINATES[new_state-1][0]), max(STATES_TO_COORDINATES[self.state-1][0], STATES_TO_COORDINATES[new_state-1][0]) + 1):
            for j in range(min(STATES_TO_COORDINATES[self.state-1][1], STATES_TO_COORDINATES[new_state-1][1]), max(STATES_TO_COORDINATES[self.state-1][1], STATES_TO_COORDINATES[new_state-1][1]) + 1):
                if (self.maze[j][i] == 4):
                    contains_door = True
        if contains_door:
            new_state = self.state
        reward = -1
        if new_state == self.goal_corner_state:
            print("hypothetical goal reached")
            reward = 50
        return reward, new_state  

    def move(self, move):
        moves = [0,1,2,3]
        self.state = self.identify_state(self.x, self.y)

        # check if new state is valid
        new_state = LEGAL_MOVEMENT[self.state-1][moves[move]] if LEGAL_MOVEMENT[self.state-1][moves[move]] != 0 else self.state

        # checks if there is a door within the rectangle formed with the corners of new and current state
        contains_door = False
        for i in range(min(STATES_TO_COORDINATES[self.state-1][0], STATES_TO_COORDINATES[new_state-1][0]), max(STATES_TO_COORDINATES[self.state-1][0], STATES_TO_COORDINATES[new_state-1][0]) + 1):
            for j in range(min(STATES_TO_COORDINATES[self.state-1][1], STATES_TO_COORDINATES[new_state-1][1]), max(STATES_TO_COORDINATES[self.state-1][1], STATES_TO_COORDINATES[new_state-1][1]) + 1):
                if (self.maze[j][i] == 4):
                    contains_door = True
        if contains_door:
            new_state = self.state
        self.x = STATES_TO_COORDINATES[new_state-1][0]
        self.y = STATES_TO_COORDINATES[new_state-1][1]
        self.state = new_state
        reward = -1
        if self.state == self.goal_corner_state:
            print("goal reached")
            self.start = random.choice(self.start_options)
            self.write_open_tiles()
            self.configure_doors_for_goal_seeking()
            cell = self.identify_start_cell(self.start)
            self.y = cell[1]
            self.x = cell[0]
            self.state = self.identify_state(self.x, self.y)
            reward = 50
        self._infer_orientation()
        return reward, new_state            


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
        self.display_text = {}
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
    
        # text = font.render(f"Location: ({x}, {y})   State: currently undefined", True, self.WHITE)
        self.display_text.update({"Location ": f"({x}, {y})"})
        
        
        # self.screen.blit(text, (300, 10))
        pygame.draw.polygon(self.screen, self.RED, [direction_points[orientation % 4], direction_points[(orientation+1)%4],direction_points[4+(orientation)%4]])

    def draw_text(self):
        final_render = ""
        for key in self.display_text:
            final_render = final_render + (f"{key}: {self.display_text[key]}  ")
        font = pygame.font.SysFont(None, 30)
        text = font.render(final_render, True, self.WHITE)
        self.screen.blit(text, (10, 10))

    def draw(self, env):
        maze = env.maze
        self.draw_maze(maze, env.maze_width, env.maze_height)
        self.draw_agent(env.x, env.y, env.orientation)
        self.draw_available_states(env.state)
        self.display_text.update({"State": f"{env.state}"})
        self.draw_text()

    def draw_available_states(self, state):
        available_states = LEGAL_MOVEMENT[state-1]
        for i in range(4):
            if available_states[i] != 0:
                pygame.draw.circle(self.screen, self.LIGHT_RED, (STATES_TO_COORDINATES[available_states[i]-1][0]*self.cell_size + (self.cell_size //2), STATES_TO_COORDINATES[available_states[i]-1][1]*self.cell_size + (self.cell_size //2),), 10)

    def run(self, environment, q_model):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        prediction = q_model.choose_action(environment.state)
                        self.display_text.update({"predicted state": f"{prediction}"})
                        q_model.step(environment)
                    # print(model.choose_action(STATES_TO_COORDINATES.index((agent.y, agent.x))))
                    if event.key == pygame.K_UP:
                        environment.move(0)
                    elif event.key == pygame.K_DOWN:
                        environment.move(3)
                    elif event.key == pygame.K_LEFT:
                        environment.move(1)
                    elif event.key == pygame.K_RIGHT:
                        environment.move(2)
            self.draw(environment)
            pygame.display.flip()
        
def main():
    screen = Screen()
    maze_width = screen.width//screen.cell_size
    maze_height = (screen.height-50)//screen.cell_size
    environment = Environment(maze_width, maze_height, start = [0,2])
    q_model = QLearningModel()
    screen.run(environment, q_model)

    pygame.quit()
    


if __name__ == "__main__":
    main()