import pygame
import random

pygame.init()

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 650
CELL_SIZE = 20
MAZE_WIDTH  = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT-50) // CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)


def create_maze():
  maze = [[0] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
  for _ in range(200):
    x = random.randint(0, MAZE_WIDTH-1)
    y = random.randint(0, MAZE_HEIGHT-1)
    maze[y][x] = 1
  maze[MAZE_HEIGHT - 1][MAZE_WIDTH - 1] = 2
  return maze

def draw_maze(screen, maze):
  for y in range(MAZE_HEIGHT):
    for x in range(MAZE_WIDTH):
      if maze[y][x] == 1:
        pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
      elif maze[y][x] == 2:
        pygame.draw.rect(screen, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

class Player:
  def __init__(self):
    self.x = 0
    self.y = 0
  def move(self, dx, dy, maze):
    new_x = self.x + dx
    new_y = self.y + dy
    if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and maze[new_y]:
      self.x = new_x
      self.y = new_y
  def draw(self, screen):
    pygame.draw.rect(screen, GREEN, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

class Timer:
  def __init__(self, countdown_time):
    self.font = pygame.font.SysFont(None, 30)
    self.start_time = pygame.time.get_ticks()
    self.countdown_time = countdown_time
  def get_time(self):
    elapsed_time = pygame.time.get_ticks() - self.start_time
    remaining_time = max(0, self.countdown_time - elapsed_time // 1000)
    if remaining_time < 0:
      remaining_time = 0
    minutes = remaining_time // 60
    seconds = remaining_time % 60
    return f"Time: {minutes:02}:{seconds:02}"
  def draw(self, screen):
    time_text = self.font.render(self.get_time(), True, BLACK)
    screen.blit(time_text, (10, 600))
  def is_time_up(self):
    elapsed_time = pygame.time.get_ticks() - self.start_time
    remaining_time = self.countdown_time - elapsed_time // 1000
    return remaining_time <= 0
  
def main():
  screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
  font = pygame.font.SysFont(None, 30)
  pygame.display.set_caption("Maze Game")
  clock = pygame.time.Clock()
  maze = create_maze()
  player = Player()
  countdown_time = 120
  timer = Timer(countdown_time)
  running = True
  won = False
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
          player.move(0, -1, maze)
        elif event.key== pygame.K_DOWN:
          player.move(0,1,maze)
        elif event.key == pygame.K_LEFT:
          player.move(-1, 0, maze)
        elif event.key == pygame.K_RIGHT:
          player.move(1,0, maze)
    screen.fill(WHITE)
    draw_maze(screen, maze)
    player.draw(screen)
    timer.draw(screen)
    if maze[player.y][player.x] == 2:
      won = True
      running = False
    if timer.is_time_up():
      running = False
    pygame.display.flip()
    clock.tick(30)
  screen.fill(WHITE)
  if won:
    time_text = font.render('You won!', True, BLACK)
  else:
      time_text = font.render('Time is up!', True, BLACK)
  screen.blit(time_text, (SCREEN_WIDTH // 2 - time_text.get_width() // 2, SCREEN_HEIGHT // 2 - time_text.get_height() // 2))
  pygame.display.flip()
  pygame.time.wait(3000)
  pygame.quit()
if __name__ == "__main__":
  main()