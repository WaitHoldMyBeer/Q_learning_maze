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

    def __init__(self,
                 width=7,
                 height=7,
                 cue_on=True,
                 goal_corner=(0, 0),
                 proximal_start_arm=(3, 0),
                 distal_start_arm=(3, 6),
                 max_env_steps=2000,
                 reward_shaping=True,
                 stagnation_window=50,  # Length of position history to track
                 stagnation_area_size=3,  # Size of area (3x3) to check for stagnation
                 stagnation_penalty=-0.2):  # Penalty for staying in same area
        super(ComplexMazeEnv, self).__init__()

        self.width = width
        self.height = height
        self.max_env_steps = max_env_steps
        self.reward_shaping = reward_shaping

        # New parameters for stagnation detection
        self.stagnation_window = stagnation_window
        self.stagnation_area_size = stagnation_area_size
        self.stagnation_penalty = stagnation_penalty
        self.position_history = deque(maxlen=stagnation_window)

        self.cue_on = cue_on
        self.cue_bit = 1 if cue_on else 0

        self.all_corners = [
            (0, 0),
            (0, width - 1),
            (height - 1, 0),
            (height - 1, width - 1)
        ]
        self.goal_corner = goal_corner
        self.other_corners = [c for c in self.all_corners if c != self.goal_corner]

        self.proximal_start_arm = proximal_start_arm
        self.distal_start_arm = distal_start_arm

        obs_low = np.array([0, 0, 0, 0], dtype=np.int32)
        obs_high = np.array([height - 1, width - 1, 3, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(4,), dtype=np.int32)

        self.action_space = spaces.Discrete(4)

        self.num_correct_trials = 0
        self.num_total_trials = 0

        self.in_guided_mode = False
        self.next_start_arm = None

        self.env_step_count = 0
        self.pending_reward = 0.0

        self.open_cells = set()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env_step_count = 0
        self.num_correct_trials = 0
        self.num_total_trials = 0
        self.pending_reward = 0.0
        
        # Reset position history for stagnation detection
        self.position_history = deque(maxlen=self.stagnation_window)

        self.random_start = random.choice([self.proximal_start_arm, self.distal_start_arm])
        self._start_new_trial(self.random_start)
        return self._get_obs(), {}

    def _start_new_trial(self, start_arm):
        self.agent_x, self.agent_y = start_arm
        self.agent_orientation = self._infer_orientation(start_arm)
        self.in_guided_mode = False
        self.pending_reward = 0.0
        self._configure_doors_for_goal_seeking()

    def _infer_orientation(self, cell):
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
        self.open_cells = self._build_plus_and_perimeter()

    def _configure_doors_for_guidance(self, next_arm):
        current_corner = (self.agent_x, self.agent_y)
        path_on_perimeter = self._perimeter_path(current_corner, next_arm)
        self.open_cells = set(path_on_perimeter)

    def _build_plus_and_perimeter(self):
        cells = set()
        for col in range(self.width):
            cells.add((0, col))
            cells.add((self.height - 1, col))
        for row in range(self.height):
            cells.add((row, 0))
            cells.add((row, self.width - 1))
        mid_row = self.height // 2
        mid_col = self.width // 2
        for c in range(self.width):
            cells.add((mid_row, c))
        for r in range(self.height):
            cells.add((r, mid_col))
        return cells

    def _perimeter_path(self, start_cell, goal_cell):
        perimeter = set()
        for col in range(self.width):
            perimeter.add((0, col))
            perimeter.add((self.height - 1, col))
        for row in range(self.height):
            perimeter.add((row, 0))
            perimeter.add((row, self.width - 1))
        if start_cell not in perimeter:
            perimeter.add(start_cell)
        if goal_cell not in perimeter:
            perimeter.add(goal_cell)
        queue = deque([start_cell])
        visited = {start_cell}
        came_from = {}
        def neighbors(x, y):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in perimeter:
                    yield (nx, ny)
        found_path = False
        while queue:
            current = queue.popleft()
            if current == goal_cell:
                found_path = True
                break
            for nxt in neighbors(*current):
                if nxt not in visited:
                    visited.add(nxt)
                    came_from[nxt] = current
                    queue.append(nxt)
        if not found_path:
            return [start_cell]
        path = []
        cur = goal_cell
        while cur != start_cell:
            path.append(cur)
            cur = came_from[cur]
        path.append(start_cell)
        path.reverse()
        return path

    def _choose_next_start_arm(self):
        return random.choice([self.proximal_start_arm, self.distal_start_arm])

    def _get_obs(self):
        return np.array([self.agent_x, self.agent_y, self.agent_orientation, self.cue_bit], dtype=np.int32)

    def step(self, action):
        self.env_step_count += 1
        done = False
        info = {}

        reward = self.pending_reward
        self.pending_reward = 0.0

        if self.reward_shaping and not self.in_guided_mode:
            old_distance = abs(self.agent_x - self.goal_corner[0]) + abs(self.agent_y - self.goal_corner[1])
        else:
            old_distance = None

        if action == 0:
            self._attempt_move_forward()
        elif action == 1:
            self.agent_orientation = (self.agent_orientation - 1) % 4
        elif action == 2:
            self.agent_orientation = (self.agent_orientation + 1) % 4
        elif action == 3:
            self.agent_orientation = (self.agent_orientation + 2) % 4

        # Record agent's position for stagnation detection
        self.position_history.append((self.agent_x, self.agent_y))
        
        if not self.in_guided_mode:
            if (self.agent_x, self.agent_y) == self.goal_corner:
                self._finish_trial(correct=True)
                done = True
            elif (self.agent_x, self.agent_y) in self.other_corners:
                self._finish_trial(correct=False)
                done = True
        else:
            if (self.agent_x, self.agent_y) == self.next_start_arm:
                done = True

        if self.reward_shaping and not self.in_guided_mode:
            if action == 0 and old_distance is not None:
                new_distance = abs(self.agent_x - self.goal_corner[0]) + abs(self.agent_y - self.goal_corner[1])
                reward += (old_distance - new_distance) * 0.1
            else:
                reward += -0.01
                
            # Check for stagnation and apply penalty if needed
            if len(self.position_history) >= self.stagnation_window:
                if self._is_stagnating():
                    reward += self.stagnation_penalty
                    info['stagnating'] = True

        if self.env_step_count >= self.max_env_steps:
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, info
        
    def _is_stagnating(self):
        """Check if the agent is staying within a small area (stagnating)"""
        if len(self.position_history) < self.stagnation_window:
            return False
            
        # Find the bounding box of recent positions
        x_coords = [pos[0] for pos in self.position_history]
        y_coords = [pos[1] for pos in self.position_history]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Check if bounding box is smaller than or equal to stagnation_area_size x stagnation_area_size
        return (max_x - min_x + 1 <= self.stagnation_area_size and 
                max_y - min_y + 1 <= self.stagnation_area_size)

    def _attempt_move_forward(self):
        x, y = self.agent_x, self.agent_y
        if self.agent_orientation == 0:   
            nx, ny = x - 1, y
        elif self.agent_orientation == 1: 
            nx, ny = x, y + 1
        elif self.agent_orientation == 2: 
            nx, ny = x + 1, y
        else:                             
            nx, ny = x, y - 1
        if (nx, ny) in self.open_cells:
            self.agent_x, self.agent_y = nx, ny

    def _finish_trial(self, correct):
        self.num_total_trials += 1
        if correct:
            self.num_correct_trials += 1
            self.pending_reward = 1.0
        else:
            self.pending_reward = -1.0
        self.in_guided_mode = True
        self.next_start_arm = self._choose_next_start_arm()
        self._configure_doors_for_guidance(self.next_start_arm)

    def render(self, mode="human"):
        pass

    def close(self):
        pass

# Helper functions for turn evaluation
def compute_desired_orientation(agent_x, agent_y, goal_corner):
    # A simple heuristic: choose the axis with greater distance
    dx = goal_corner[0] - agent_x
    dy = goal_corner[1] - agent_y
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 0  # 2: south, 0: north
    else:
        return 1 if dy > 0 else 3  # 1: east, 3: west

def angular_distance(a, b):
    diff = abs(a - b)
    return min(diff, 4 - diff)