import pygame
import random
import numpy as np

pygame.init()

SCREEN_WIDTH = 810 # from here
SCREEN_HEIGHT = 860
CELL_SIZE = 90
MAZE_WIDTH  = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT-50) // CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
LIGHT_RED = (255, 100, 100) # to here is accoutned for 

STATES_TO_COORDINATES = [ # y 0 is at top
      (4,3), (4,2), (4,5), (4,6), (3,4), (2,4), (5,4), (6,4), #center plus
      (3,1), (2,1), (5,1), (6,1), # top perimeter
      (3,7), (2,7), (5,7), (6,7), # bottom perimeter
      (1, 3), (1,2), (1,5), (1,6), #left perimeter
      (7,3), (7,2), (7,5), (7,6), # right perimeter
      (7,1), (8,0), # top right
      (7,7), (8,8), # bottom right corner
      (1,7), (0,8), # bottom left corner 
      (1,1), (0,0) # top left corner
]
BLUE = (0, 0, 255)
def find_start_cell(start):
    if start == 0:
      return (MAZE_HEIGHT//2, MAZE_HEIGHT//2 - 1)
    elif start == 1:
      return (MAZE_HEIGHT//2 + 1, MAZE_WIDTH//2)
    elif start == 2:
      return (MAZE_HEIGHT//2, MAZE_WIDTH//2 + 1)
    elif start == 3:
      return (MAZE_HEIGHT//2 - 1, MAZE_WIDTH//2)

# (y inverse, x)
class Maze:
  def __init__(self, goal_corner = 0, start = 0):
    self.maze = [[0] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
    self.start = start
    for x in range(1,MAZE_WIDTH-1):
      self.maze[1][x] = 1
      self.maze[MAZE_HEIGHT-2][x] = 1
      self.maze[MAZE_HEIGHT//2][x] = 1
    for y in range(1,MAZE_HEIGHT-1):
      self.maze[y][MAZE_WIDTH-2] = 1
      self.maze[y][MAZE_WIDTH//2] = 1
      self.maze[y][1] = 1
    self.maze[0][0] = 2 if goal_corner == 0 else 3
    self.maze[0][MAZE_WIDTH-1] = 2 if goal_corner == 1 else 3
    self.maze[MAZE_HEIGHT-1][0] = 2 if goal_corner == 3 else 3
    self.maze[MAZE_HEIGHT-1][MAZE_WIDTH-1] = 2 if goal_corner == 2 else 3
    self.trials = 0
    self.blocks = 0
  def draw_maze(self, screen):
    for y in range(MAZE_HEIGHT):
      for x in range(MAZE_WIDTH):
        if self.maze[y][x] == 1: # empty space
          pygame.draw.rect(screen, GRAY, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        elif self.maze[y][x] == 0: # wall
          pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        elif self.maze[y][x] == 2: # goal
          pygame.draw.rect(screen, GREEN, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        elif self.maze[y][x] == 3: # empty well
          pygame.draw.rect(screen, WHITE, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        elif self.maze[y][x] == 4: # door
          pygame.draw.rect(screen, BLUE, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
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
  def configure_doors_for_return(self):
    
    if self.start == 0:
      self.maze[STATES_TO_COORDINATES[self.potential_return_doors[3]-1][1]][STATES_TO_COORDINATES[self.potential_return_doors[3]-1][0]] = 4
  def start_trial(self):
    self.start = random.choice([0, 1, 2, 3])
    self.configure_doors_for_goal_seeking()
    return self.start

#update for direction and restricted movement
class Agent:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.orientation = 0 #0 north, 1 east, 2 south, 3 west
    
    self.states_to_orientations = [2, 0, 0, 2, 1, 3, 3, 1, # center plus
      1, 3, 3, 1, # top perimeter
      1, 3, 3, 1, # bottom perimeter
      2, 0, 0, 2, #left perimeter
      2, 0, 0, 2, # right perimeter
      2, 0, # top right
      0, 2, # bottom right corner
      0, 2, # bottom left corner
      2, 0 # top left corner
    ]
    self.legal_movement = [ # forward, left, right, about face | 0 is not allowed
      [4, 8, 6, 2],
      [0, 10, 12, 1],
      [2, 6, 8, 4],
      [0, 16, 14, 3], #4
      [8, 2, 4, 6],
      [0, 20, 18, 5],
      [6, 4, 2, 8],
      [0, 22, 24, 7], #8
      [12, 0, 1, 10], #9
      [0, 17, 32, 9],
      [10, 1, 0, 12], #11
      [0, 26, 21, 11], # 12
      [16, 3, 0, 14],
      [0, 30, 19, 13],
      [14, 0, 3, 16],
      [0, 23, 28, 15], #16
      [20, 5, 0, 18],
      [0, 32, 9, 17],
      [18, 0, 5, 20],
      [0, 13, 30, 19], #20
      [24, 0, 7, 22],
      [0, 11, 26, 21],
      [22, 7, 0, 24],
      [0, 28, 15, 23], #24
      [0, 21, 11, 26],
      [0, 0, 0, 25],
      [0, 15, 23, 28],
      [0, 0, 0, 27], #28
      [0, 19, 13, 30],
      [0, 0, 0, 29],
      [0, 9, 17, 32],
      [0, 0, 0, 31], #32
    ]
    self.state = STATES_TO_COORDINATES.index((self.x, self.y)) + 1 if ((self.x, self.y) in STATES_TO_COORDINATES) else 0
  def move(self, move, maze):
    moves = [0,1,2,3]
    current_state = STATES_TO_COORDINATES.index((self.x, self.y)) + 1 if ((self.x, self.y) in STATES_TO_COORDINATES) else -1
    new_state = self.legal_movement[current_state-1][moves[move]] if self.legal_movement[current_state-1][moves[move]] != 0 else current_state

    contains_door = False
    for i in range(min(STATES_TO_COORDINATES[current_state-1][0], STATES_TO_COORDINATES[new_state-1][0]), max(STATES_TO_COORDINATES[current_state-1][0], STATES_TO_COORDINATES[new_state-1][0]) + 1):
      for j in range(min(STATES_TO_COORDINATES[current_state-1][1], STATES_TO_COORDINATES[new_state-1][1]), max(STATES_TO_COORDINATES[current_state-1][1], STATES_TO_COORDINATES[new_state-1][1]) + 1):

        if (maze.maze[j][i] == 4):
          contains_door = True
    if contains_door:
      new_state = current_state
    self.x = STATES_TO_COORDINATES[new_state-1][0]
    self.y = STATES_TO_COORDINATES[new_state-1][1]
    self.state = new_state
    """movement = [(0, -1), (-1,0), (0,1),(1, 0)]
    new_x = self.x + movement[self.orientation][0]
    new_y = self.y + movement[self.orientation][1]
    if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and maze[new_y][new_x] != 0:
      self.x = new_x
      self.y = new_y"""
  def _infer_orientation(self): # left = 0, right = 1
    self.orientation = self.states_to_orientations[self.state-1]
  def draw(self, screen):
    #Direction Array
    direction_points = [
        (self.x * CELL_SIZE + CELL_SIZE, self.y * CELL_SIZE + CELL_SIZE),  # bottom right
        (self.x * CELL_SIZE, self.y * CELL_SIZE + CELL_SIZE),  # bottom left
        (self.x * CELL_SIZE, self.y * CELL_SIZE),  # top left
        (self.x * CELL_SIZE + CELL_SIZE, self.y * CELL_SIZE),  # top right
        (self.x * CELL_SIZE + (CELL_SIZE // 2), self.y * CELL_SIZE),  # top middle
        (self.x * CELL_SIZE + CELL_SIZE, self.y * CELL_SIZE + (CELL_SIZE // 2)),  # right middle
        (self.x * CELL_SIZE + (CELL_SIZE // 2), self.y * CELL_SIZE + CELL_SIZE),  # bottom middle
        (self.x * CELL_SIZE, self.y * CELL_SIZE + (CELL_SIZE // 2)),  # left middle
    ]
    font = pygame.font.SysFont(None, 30)
    
    available_states = self.legal_movement[self.state-1]
    for i in range(4):
      if available_states[i] != 0:
        pygame.draw.circle(screen, LIGHT_RED, (STATES_TO_COORDINATES[available_states[i]-1][0]*CELL_SIZE + (CELL_SIZE //2), STATES_TO_COORDINATES[available_states[i]-1][1]*CELL_SIZE + (CELL_SIZE //2),), 10)
    self._infer_orientation()
    text = font.render(f"Location: ({self.x}, {self.y})   State: {self.state}", True, WHITE)
    screen.blit(text, (300, 10))
    pygame.draw.polygon(screen, RED, [direction_points[self.orientation % 4], direction_points[(self.orientation+1)%4], direction_points[4+(self.orientation)%4]])
  
  def reset(self):
    self.start = random.choice([0, 1, 2, 3])
    cell = find_start_cell(self.start)
    self.y = cell[0]
    self.x = cell[1]

class Reward:
  def __init__(self, cumulative_reward, current_reward):
    self.cumulative_reward = cumulative_reward
    self.current_reward = current_reward
  def update(self, reward):
    self.cumulative_reward += reward
    self.current_reward = reward
  def draw(self, screen):
    font = pygame.font.SysFont(None, 30)
    text = font.render(f"Reward: {self.cumulative_reward}", True, WHITE)
    screen.blit(text, (10, 10))
  def reset(self):
    self.current_reward = 0


class QLearningModel:
  def __init__(self, n_states = 32, n_actions = 4, alpha = 0.9, # learning rate
    gamma = 0.95, # discount factor
    epsilon = 1.0, # exploration rate
    epsilon_decay = 0.9995, # decay rate for epsilon
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
    self.epsilon_decay = 0.9995
    self.min_epsilon = min_epsilon
    self.block_size = block_size
    self.max_steps = max_steps
    self.discount_factor = gamma
  def choose_action(self, state):
    if random.uniform(0,1) < self.exploration_probability:
      return random.choice(self.action_space)
    else:
      return np.argmax(self.q_table[state])
  def step(self, old_state, new_state, reward):
    action = self.choose_action(state)
    old_value = self.q_table[old_state, action]
    next_max = np.max(self.q_table[new_state, :])
    self.q_table[state, action] = (1-self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
    state = new_state
    return 

def main():
  screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
  pygame.display.set_caption("Maze Game")
  maze = Maze(0, 3)
  start = maze.start_trial()
  maze.draw_maze(screen)
  agent = Agent(find_start_cell(start)[0], find_start_cell(start)[1])
  reward = Reward(0, 0)
  model = QLearningModel()
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
    screen.fill(WHITE)
    maze.draw_maze(screen)
    agent.draw(screen)
    reward.draw(screen)
    if maze.maze[agent.y][agent.x] == 2:
      reward.update(10)
      agent.reset()
      maze.start_trial()
    pygame.display.flip()

if __name__ == "__main__":
  main()