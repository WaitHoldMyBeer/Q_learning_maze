import pygame
import random

pygame.init()

SCREEN_WIDTH = 810
SCREEN_HEIGHT = 860
CELL_SIZE = 90
MAZE_WIDTH  = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT-50) // CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

# (y inverse, x)
def create_maze():
  maze = [[0] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
  for x in range(1,MAZE_WIDTH-1):
    maze[1][x] = 1
    maze[MAZE_HEIGHT-2][x] = 1
    maze[MAZE_HEIGHT//2][x] = 1
  for y in range(1,MAZE_HEIGHT-1):
    maze[y][MAZE_WIDTH-2] = 1
    maze[y][MAZE_WIDTH//2] = 1
    maze[y][1] = 1
  maze[0][0] = 2
  maze[0][MAZE_WIDTH-1] = 3
  maze[MAZE_HEIGHT-1][0] = 3
  maze[MAZE_HEIGHT-1][MAZE_WIDTH-1] = 3
  return maze

def draw_maze(screen, maze):
  for y in range(MAZE_HEIGHT):
    for x in range(MAZE_WIDTH):
      if maze[y][x] == 1:
        pygame.draw.rect(screen, GRAY, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
      elif maze[y][x] == 0:
        pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
      elif maze[y][x] == 2:
        pygame.draw.rect(screen, GREEN, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
      elif maze[y][x] == 3:
        pygame.draw.rect(screen, WHITE, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

#update for direction and restricted movement
class Agent:
  def __init__(self, x, y):
    self.x = x
    self.y = y
  def move(self, dx, dy, maze):
    new_x = self.x + dx
    new_y = self.y + dy
    if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and maze[new_y][new_x] != 0:
      self.x = new_x
      self.y = new_y
  def draw(self, screen):
    pygame.draw.rect(screen, RED, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
  def reset(self):
    self.x = MAZE_HEIGHT//2
    self.y = MAZE_WIDTH//2

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

def main():
  screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
  font = pygame.font.SysFont(None, 30)
  pygame.display.set_caption("Maze Game")
  clock = pygame.time.Clock()
  maze = create_maze()
  agent = Agent(MAZE_HEIGHT//2, MAZE_WIDTH//2)
  reward = Reward(0, 0)
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
          agent.move(0, -1, maze)
        elif event.key== pygame.K_DOWN:
          agent.move(0,1,maze)
        elif event.key == pygame.K_LEFT:
          agent.move(-1, 0, maze)
        elif event.key == pygame.K_RIGHT:
          agent.move(1,0, maze)
    screen.fill(WHITE)
    draw_maze(screen, maze)
    agent.draw(screen)
    reward.draw(screen)
    if maze[agent.y][agent.x] == 2:
      reward.update(10)
      agent.reset()
    pygame.display.flip()

if __name__ == "__main__":
  main()