import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from TwoArmedMazeEnv import compute_desired_orientation, angular_distance

class QLearningModel:
    def __init__(self, env,
                                num_episodes=100,
                                alpha=0.1,
                                gamma=0.9,
                                initial_epsilon=1.0,
                                min_epsilon=0.1,
                                epsilon_decay=0.99):
        """
        Q-learning routine that logs multiple metrics per episode.
        Also tracks the fraction of correct first and second turns.
        """
        self.env = env
        self.num_episodes=num_episodes
        self.alpha=alpha
        self.gamma=gamma
        self.initial_epsilon=initial_epsilon
        self.min_epsilon=min_epsilon
        self.epsilon_decay=epsilon_decay
        self.q_table = np.zeros((env.height, env.width, 4, 2, 4))
        self.metrics = {
            'fraction_correct': [],
            'cumulative_reward': [],
            'steps_per_trial': [],
            'epsilon': [],
            'first_turn_fraction': [],
            'second_turn_fraction': [],
            'first_corner': [],
            'start_arm': [],
        }
        self.epsilon = initial_epsilon
        
        # Initialize metrics for each episode
        self.reset_episode_metrics()
    
    def reset_episode_metrics(self):
        # For tracking turning decisions across trials within an episode:
        self.first_turn_correct_total = 0
        self.first_turn_total = 0
        self.second_turn_correct_total = 0
        self.second_turn_total = 0
        self.trial_turn_count = 0  # counts turns in current trial
        self.last_trial_total = 0  # to detect trial boundaries
        self.cumulative_reward = 0.0
        self.steps = 0
        self.corner = None  # Keep track of first corner encountered

    def reset_block_metrics(self):
        """Reset only the metrics for a new block, preserving the Q-table and epsilon."""
        self.metrics = {
            'fraction_correct': [],
            'cumulative_reward': [],
            'steps_per_trial': [],
            'epsilon': [],
            'first_turn_fraction': [],
            'second_turn_fraction': [],
            'first_corner': [],
            'start_arm': [],
        }
        
    def run_q_learning(self):
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            done = False
            
            # Reset episode metrics
            self.reset_episode_metrics()
            self.last_trial_total = self.env.num_total_trials
            check_first_corner = True

            while not done:
                state = tuple(obs)  # (x, y, orientation, cue_bit)
                
                # Record first corner encountered
                if ((self.env.agent_x, self.env.agent_y) in self.env.all_corners and check_first_corner):
                    self.corner = self.env.all_corners.index((self.env.agent_x, self.env.agent_y))
                    check_first_corner = False
                
                # Decide on action (epsilon-greedy)
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])
                    
                # --- Turn tracking ---
                # We record only if the action is a turning action (left or right)
                if action in [1, 2]:
                    old_orientation = obs[2]
                    desired = compute_desired_orientation(obs[0], obs[1], self.env.goal_corner)
                    if action == 1:
                        new_orientation = (old_orientation - 1) % 4
                    elif action == 2:
                        new_orientation = (old_orientation + 1) % 4
                    diff_old = angular_distance(old_orientation, desired)
                    diff_new = angular_distance(new_orientation, desired)
                    is_correct = diff_new < diff_old  # turn is "correct" if it brings agent closer

                    # Record first turn if not yet recorded in this trial
                    if self.trial_turn_count == 0:
                        self.first_turn_total += 1
                        if is_correct:
                            self.first_turn_correct_total += 1
                        self.trial_turn_count += 1
                    # Else, if this is the second turn in this trial (only record the first two turns)
                    elif self.trial_turn_count == 1:
                        self.second_turn_total += 1
                        if is_correct:
                            self.second_turn_correct_total += 1
                        self.trial_turn_count += 1
                    # If more than 2 turns, we ignore for this analysis.
                # --- End Turn tracking ---

                next_obs, reward, done, _, _ = self.env.step(action)
                
                self.cumulative_reward += reward
                self.steps += 1

                next_state = tuple(next_obs)
                # Update Q-table
                old_val = self.q_table[state][action]  # Corrected indexing
                next_max = np.max(self.q_table[next_state]) if not done else 0.0
                td_target = reward + self.gamma * next_max
                self.q_table[state][action] = old_val + self.alpha * (td_target - old_val)  # Corrected indexing

                obs = next_obs

                # Check if a trial boundary occurred: env.num_total_trials increased
                if self.env.num_total_trials > self.last_trial_total:
                    # A new trial has started, so reset the per-trial turn counter.
                    self.trial_turn_count = 0
                    self.last_trial_total = self.env.num_total_trials
            
            # Update metrics after each episode
            self.update_metrics()
            
            # Print information about the episode
            corner_name = "None"
            if self.corner is not None:
                corner_positions = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
                corner_name = corner_positions[self.corner]
                
            # Print number of trials, steps, and first corner
            print(f"Episode {episode+1}: Steps: {self.steps}, First Corner: {corner_name}, " +
                  f"Trials: {self.env.num_total_trials}, Correct trials: {self.env.num_correct_trials}")
            
            # Print if goal was reached (fraction = 1.0 means all trials were correct)
            fraction_correct = self.metrics['fraction_correct'][-1]
            if fraction_correct == 1.0:
                print(f"  ✓ All trials correct! Reward: {self.cumulative_reward:.2f}")
            else:
                print(f"  ✗ Success rate: {fraction_correct:.2f}, Reward: {self.cumulative_reward:.2f}")
                    
    def update_metrics(self):
        total_trials = self.env.num_total_trials
        correct_trials = self.env.num_correct_trials
        fraction = correct_trials / total_trials if total_trials > 0 else 0.0
        steps_per_trial = self.steps / total_trials if total_trials > 0 else 0.0

        # Compute fraction of correct first and second turns for this episode
        first_turn_fraction = self.first_turn_correct_total / self.first_turn_total if self.first_turn_total > 0 else 0.0
        second_turn_fraction = self.second_turn_correct_total / self.second_turn_total if self.second_turn_total > 0 else 0.0

        self.metrics['fraction_correct'].append(fraction)
        self.metrics['cumulative_reward'].append(self.cumulative_reward)
        self.metrics['steps_per_trial'].append(steps_per_trial)
        self.metrics['epsilon'].append(self.epsilon)
        self.metrics['first_turn_fraction'].append(first_turn_fraction)
        self.metrics['second_turn_fraction'].append(second_turn_fraction)
        self.metrics['first_corner'].append(self.corner if self.corner is not None else -1)
        self.metrics['start_arm'].append(1 if tuple(self.env.random_start) == tuple(self.env.distal_start_arm) else 0)
        
        # Update epsilon for next episode
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)