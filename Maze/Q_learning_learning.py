import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

class ComplexMazeEnv(gym.Env):
    """
    A more complex environment reflecting the real experiment’s logic.
    It features:
      - Four possible actions: move forward, turn left, turn right, turn around.
      - Door configurations for goal-seeking vs. guided mode.
      - Multiple sequential trials within a single episode.
      - Optional reward shaping to give intermediate feedback.
    """
    metadata = {"render_modes": ["human"]}

    #create self object, width, height, cue, goal corner, proximal_start_arm, distal_start_arm, max steps you can take, and the reward shaping
    def __init__(self,
                 width=7,
                 height=7,
                 cue_on=True,
                 goal_corner=(0, 0),
                 proximal_start_arm=(3, 0),
                 distal_start_arm=(3, 6),
                 max_env_steps=2000,
                 reward_shaping=True):
        #initializes to the upper class
        super(ComplexMazeEnv, self).__init__()

        self.width = width
        self.height = height
        self.max_env_steps = max_env_steps
        self.reward_shaping = reward_shaping

        #set cue bit
        self.cue_on = cue_on
        self.cue_bit = 1 if cue_on else 0

        #set goal corner
        all_corners = [
            (0, 0),
            (0, width - 1),
            (height - 1, 0),
            (height - 1, width - 1)
        ]
        self.goal_corner = goal_corner
        self.other_corners = [c for c in all_corners if c != self.goal_corner]

        #set start rm
        self.proximal_start_arm = proximal_start_arm
        self.distal_start_arm = distal_start_arm

        #set observation space
        obs_low = np.array([0, 0, 0, 0], dtype=np.int32)
        obs_high = np.array([height - 1, width - 1, 3, 1], dtype=np.int32)
        #uses increments of integers from low observation array to upper observation array, setting the shape of this space as 4
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(4,), dtype=np.int32)

        #we can take 4 actions
        self.action_space = spaces.Discrete(4)

        #counting for data
        self.num_correct_trials = 0
        self.num_total_trials = 0

        #movement info
        self.in_guided_mode = False
        self.next_start_arm = None

        #algo info
        self.env_step_count = 0
        self.pending_reward = 0.0

        #initializes cells visitable
        self.open_cells = set()

        self.reset()

    #resets the environment varialbes and puts in new start arm
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env_step_count = 0
        self.num_correct_trials = 0
        self.num_total_trials = 0
        self.pending_reward = 0.0

        #new start arm
        random_start = random.choice([self.proximal_start_arm, self.distal_start_arm])
        self._start_new_trial(random_start)
        #defined below
        return self._get_obs(), {}


    def _start_new_trial(self, start_arm):
        """Place the agent in the chosen start arm with an appropriate orientation."""
        self.agent_x, self.agent_y = start_arm
        self.agent_orientation = self._infer_orientation(start_arm)
        self.in_guided_mode = False
        self.pending_reward = 0.0
        self._configure_doors_for_goal_seeking()

    def _infer_orientation(self, cell):
        """Infer a natural orientation based on the starting cell's location."""
        x, y = cell
        if x == 0:
            return 2  
        if x == self.height - 1:
            return 0  
        if y == 0:
            return 1  
        if y == self.width - 1:
            return 3  
        return 2  

    def _configure_doors_for_goal_seeking(self):
        """Set open cells based on a central plus and perimeter pattern."""
        #opens cells to the plus and perimeter
        self.open_cells = self._build_plus_and_perimeter()

    def _configure_doors_for_guidance(self, next_arm):
        """Restrict movement along the perimeter to guide the agent back."""
        #gets current corner and returns a path with that current corner on the perimeter
        current_corner = (self.agent_x, self.agent_y)
        path_on_perimeter = self._perimeter_path(current_corner, next_arm)
        self.open_cells = set(path_on_perimeter)

    def _build_plus_and_perimeter(self):
        """Return a set of cells corresponding to the perimeter and a central plus."""
        cells = set()
        
        #loops through width indices and sets row (x axis) at top and bottom to open cells with column labels at respective columns
        for col in range(self.width):
            cells.add((0, col))
            cells.add((self.height - 1, col))
        #loops through width indices and sets columns (y axis) at top and bottom to open cells with row labels at respective columns
        for row in range(self.height):
            cells.add((row, 0))
            cells.add((row, self.width - 1))
        
        #integer division to find the middle row of the plus
        mid_row = self.height // 2
        mid_col = self.width // 2
        #repeats the same as above
        for c in range(self.width):
            cells.add((mid_row, c))
        for r in range(self.height):
            cells.add((r, mid_col))
        return cells
    
    def _perimeter_path(self, start_cell, goal_cell):
        perimeter = set()
        for col in range(self.width):
            perimeter.add(0,col)
            perimeter.add(self.height-1, col)
        for row in range(self.height):
            perimeter.add(row, 0)
            perimeter.add(row, self.width-1)
        
        if start_cell not in perimeter:
            perimeter.add(start_cell)
        if goal_cell not in perimeter:
            perimeter.add(goal_cell)
        

        queue = deque([start_cell])
        visited = {start_cell}
        came_from = {}

        def neighbors(x,y):
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if (nx,ny) in perimeter:
                    yield (nx,ny)
        
        found_path = False
        while (queue):
            current = queue.popleft()
            if current == goal_cell:
                found_path = True
                break
            for nxt in neighbors(*current):
                if nxt not in visited:
                    visited.add(nxt)
                    came_from[nxt] = current
                    queue.append(nxt)
        