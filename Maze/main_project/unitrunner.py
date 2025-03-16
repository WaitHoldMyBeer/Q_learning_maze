import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from Q_learningv1 import improved_train_q_learning
from TwoArmedMazeEnv import ComplexMazeEnv


def run_multiple_trials(num_runs=5, num_episodes=100):
    collected_metrics = {
        'fraction_correct': [],
        'cumulative_reward': [],
        'steps_per_trial': [],
        'epsilon': [],
        'first_turn_fraction': [],
        'second_turn_fraction': [],
        'first_corner': [],
        'start_arm': [],
    }
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
    return avg_metrics, std_metrics, collected_metrics

def plot_metrics(avg_metrics, std_metrics, num_episodes=1):
    episodes = np.arange(1, num_episodes + 1)
    fig, axs = plt.subplots(2, 3, figsize=(12, 10))

    # Plot overall fraction of correct trials and overlay first/second turn fractions
    axs[0, 0].errorbar(episodes, avg_metrics['fraction_correct'], yerr=std_metrics['fraction_correct'],
                       fmt='-o', capsize=3, label='Overall')
    axs[0, 0].errorbar(episodes, avg_metrics['first_turn_fraction'], yerr=std_metrics['first_turn_fraction'],
                       fmt='-s', capsize=3, label='1st Turn')
    axs[0, 0].errorbar(episodes, avg_metrics['second_turn_fraction'], yerr=std_metrics['second_turn_fraction'],
                       fmt='-^', capsize=3, label='2nd Turn')
    axs[0, 0].set_title('Fraction Correct (Overall / 1st / 2nd Turns)')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Fraction Correct')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].errorbar(episodes, avg_metrics['cumulative_reward'], yerr=std_metrics['cumulative_reward'],
                       fmt='-o', capsize=3)
    axs[0, 1].set_title('Cumulative Reward per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Cumulative Reward')
    axs[0, 1].grid(True)

    axs[1, 0].errorbar(episodes, avg_metrics['steps_per_trial'], yerr=std_metrics['steps_per_trial'],
                       fmt='-o', capsize=3)
    axs[1, 0].set_title('Steps per Trial')
    axs[1, 0].set_xlabel('Episode')
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
    NUM_EPISODES = 32
    avg_metrics, std_metrics, collected_metrics = run_multiple_trials(num_runs=NUM_RUNS, num_episodes=NUM_EPISODES)
    print(collected_metrics.get('first_corner'))
    plot_metrics(avg_metrics, std_metrics, num_episodes=NUM_EPISODES)