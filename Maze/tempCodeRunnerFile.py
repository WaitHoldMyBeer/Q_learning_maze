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

    #
    def _perimeter_path(self, start_cell, goal_cell):
        """Find a path along the perimeter cells using BFS."""
        #creates perimeter
        perimeter = set()
        for col in range(self.width):
            perimeter.add((0, col))
            perimeter.add((self.height - 1, col))
        for row in range(self.height):
            perimeter.add((row, 0))
            perimeter.add((row, self.width - 1))

        #adds start and goal cell to perimeter
        if start_cell not in perimeter:
            perimeter.add(start_cell)
        if goal_cell not in perimeter:
            perimeter.add(goal_cell)

        #creates a double ended queue storing total queue which is visited, and came from
        queue = deque([start_cell])
        visited = {start_cell}
        came_from = {}

        #basically if the neighboring cell is in the perimeter, then this function returns that cell
        def neighbors(x, y):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in perimeter:
                    yield (nx, ny)

        #run until the 
        found_path = False
        while queue:
            current = queue.popleft()
            if current == goal_cell:
                found_path = True
                break
            #*current passes in the cell as a tuple
            for nxt in neighbors(*current):
                #add all possible branches current cell
                if nxt not in visited:
                    visited.add(nxt)
                    came_from[nxt] = current
                    queue.append(nxt)

        #we stay in the current start cell
        if not found_path:
            return [start_cell]

        #set a new path to return
        path = []
        cur = goal_cell
        while cur != start_cell:
            path.append(cur)
            cur = came_from[cur]
        path.append(start_cell)
        path.reverse()
        return path

    def _choose_next_start_arm(self):
        """Randomly select the next starting arm for the guided mode."""
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

        
        if not self.in_guided_mode:
            if (self.agent_x, self.agent_y) == self.goal_corner:
                self._finish_trial(correct=True)
            elif (self.agent_x, self.agent_y) in self.other_corners:
                self._finish_trial(correct=False)
        else:
            
            if (self.agent_x, self.agent_y) == self.next_start_arm:
                self._start_new_trial(self.next_start_arm)

        
        if self.reward_shaping and not self.in_guided_mode:
            if action == 0 and old_distance is not None:
                new_distance = abs(self.agent_x - self.goal_corner[0]) + abs(self.agent_y - self.goal_corner[1])
                
                reward += (old_distance - new_distance) * 0.1
            else:
                
                reward += -0.01

        
        if self.env_step_count >= self.max_env_steps:
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, info

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

#define model
def improved_train_q_learning(env,
                              num_episodes=100,
                              alpha=0.1,
                              gamma=0.9,
                              initial_epsilon=1.0,
                              min_epsilon=0.1,
                              epsilon_decay=0.99):
    """
    A Q-learning training routine that logs multiple metrics per episode.
    Uses a refined (slower) epsilon decay and leverages reward shaping.
    """
    #build q table to state space as height, width, orientation, cue on, and 4 possible actions
    q_table = np.zeros((env.height, env.width, 4, 2, 4))

    
    metrics = {
        'fraction_correct': [],
        'cumulative_reward': [],
        'steps_per_trial': [],
        'epsilon': []
    }
    epsilon = initial_epsilon


    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        cumulative_reward = 0.0
        steps = 0

        while not done:
            
            state = tuple(obs)  
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_obs, reward, done, _, _ = env.step(action)
            cumulative_reward += reward 

            
            next_state = tuple(next_obs)
            old_val = q_table[state + (action,)]
            next_max = np.max(q_table[next_state]) if not done else 0.0
            td_target = reward + gamma * next_max
            q_table[state + (action,)] = old_val + alpha * (td_target - old_val)

            obs = next_obs
            steps += 1

        
        total_trials = env.num_total_trials
        correct_trials = env.num_correct_trials
        fraction = correct_trials / total_trials if total_trials > 0 else 0.0
        steps_per_trial = steps/total_trials if total_trials > 0 else 0.0
        metrics['fraction_correct'].append(fraction)
        metrics['cumulative_reward'].append(cumulative_reward)
        metrics['steps_per_trial'].append(steps_per_trial)
        metrics['epsilon'].append(epsilon)

        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table, metrics


def run_multiple_trials(num_runs=5, num_episodes=100):
    """
    Run multiple training sessions to compute average metrics with error bars.
    """
    
    collected_metrics = {
        'fraction_correct': [],
        'cumulative_reward': [],
        'steps_per_trial': [],
        'epsilon': []
    }

    #num runs is the average
    for run in range(num_runs):
        
        env = ComplexMazeEnv(width=7, height=7, cue_on=True, reward_shaping=True, max_env_steps=2000)
        _, metrics = improved_train_q_learning(env, num_episodes=num_episodes)
        for key in collected_metrics:
            collected_metrics[key].append(metrics[key])

    
    avg_metrics = {}
    std_metrics = {}
    for key in collected_metrics:
        data = np.array(collected_metrics[key])
        avg_metrics[key] = np.mean(data, axis=0)
        std_metrics[key] = np.std(data, axis=0)
    return avg_metrics, std_metrics


def plot_metrics(avg_metrics, std_metrics, num_episodes=100):
    """
    Create a 2x2 dashboard plot showing fraction correct, cumulative reward, steps, and epsilon per episode.
    Error bars (std deviation) are added.
    """
    episodes = np.arange(1, num_episodes + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    
    axs[0, 0].errorbar(episodes, avg_metrics['fraction_correct'], yerr=std_metrics['fraction_correct'],
                       fmt='-o', capsize=3)
    axs[0, 0].set_title('Fraction of Correct Trials')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Fraction Correct')
    axs[0, 0].grid(True)

    
    axs[0, 1].errorbar(episodes, avg_metrics['cumulative_reward'], yerr=std_metrics['cumulative_reward'],
                       fmt='-o', capsize=3)
    axs[0, 1].set_title('Cumulative Reward per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Cumulative Reward')
    axs[0, 1].grid(True)

    
    axs[1, 0].errorbar(episodes, avg_metrics['steps_per_trial'], yerr=std_metrics['steps_per_trial'],
                       fmt='-o', capsize=3)
    axs[1, 0].set_title('Steps per Trial')
    axs[1, 0].set_xlabel('Trials')
    axs[1, 0].set_ylabel('Steps')
    axs[1, 0].grid(True)

    
    axs[1, 1].errorbar(episodes, avg_metrics['epsilon'], yerr=std_metrics['epsilon'],
                       fmt='-o', capsize=3)
    axs[1, 1].set_title('Epsilon Value per Episode')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    NUM_RUNS = 5
    NUM_EPISODES = 100

    avg_metrics, std_metrics = run_multiple_trials(num_runs=NUM_RUNS, num_episodes=NUM_EPISODES)
    plot_metrics(avg_metrics, std_metrics, num_episodes=NUM_EPISODES)
