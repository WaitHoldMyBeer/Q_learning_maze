import gymnasium as gym
import numpy as np
import random

environment_states = 32

q_values = np.zeros((environment_states, 4))

actions = ['forward', 'left', 'right', 'turn']

rewards = np.full((environment_states), -1)

rewards[31] = 10

class MarkovMaze(gym.Env):
  def __init__(self):
    self.size = 32