import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from Q_learningv1 import QLearningModel
from TwoArmedMazeEnv import ComplexMazeEnv


def run_multiple_trials(num_runs=5, block_size=32, success_threshold=0.7, max_blocks=30):
    collected_metrics = {
        'fraction_correct': [],
        'cumulative_reward': [],
        'steps_per_trial': [],
        'epsilon': [],
        'first_turn_fraction': [],
        'second_turn_fraction': [],
        'first_corner': [],
        'start_arm': [],
        'blocks_to_criterion': [],
        'reached_criterion': [],
        'episode_success': [],  # NEW: per-episode binary outcomes
        'episode_start_arm': []  # NEW: track starting arm for each episode
    }
    
    for run in range(num_runs):
        # Use a more reasonable gbdegnjjfjasdjmax_env_steps value (2000 instead of 100000)
        env = ComplexMazeEnv(width=7, height=7, cue_on=True, reward_shaping=False, max_env_steps=2000)
        print(f"Run {run + 1}/{num_runs}")
        
        # Create the Q-learning model once per run instead of once per block
        q_model = QLearningModel(env, num_episodes=block_size)
        
        consecutive_success = 0
        block_count = 0
        block_metrics = {
            'fraction_correct': [],
            'cumulative_reward': [],
            'steps_per_trial': [],
            'epsilon': [],
            'first_turn_fraction': [],
            'second_turn_fraction': [],
            'first_corner': [],
            'start_arm': [],
            'blocks_to_criterion': [],
            'reached_criterion': [],
            'episode_success': []
        }
        run_episode_success = []  # store binary outcome for each episode in this run
        run_episode_start_arm = []  # store starting arm for each episode in this run

        reached_criterion = False
        while consecutive_success < 2 and block_count < max_blocks:
            # Reset the metrics before running each block (but keep the learned Q-table)
            q_model.reset_block_metrics()  # Use the new method instead of manually resetting
            
            print(f"\n=== Block {block_count + 1}/{max_blocks} ===")
            q_model.run_q_learning()
            block_count += 1
            
            # Compute block performance (average across the block)
            block_performance = np.mean(q_model.metrics['fraction_correct'])
            print(f"\nBlock {block_count} Summary:")
            print(f"  Performance: {block_performance:.2f}")
            
            # Add a summary of steps and corner visits for this block
            if 'first_corner' in q_model.metrics:
                corner_counts = {}
                for corner in q_model.metrics['first_corner']:
                    if corner == -1:
                        corner_name = "None"
                    else:
                        corner_positions = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"] 
                        corner_name = corner_positions[corner]
                    corner_counts[corner_name] = corner_counts.get(corner_name, 0) + 1
                print("  First corner visits:")
                for corner_name, count in corner_counts.items():
                    print(f"    {corner_name}: {count}")
            
            # Convert each episode's fraction_correct to a binary outcome:
            # success = 1 if fraction_correct is exactly 1.0, else 0.
            block_success = [1 if x == 1.0 else 0 for x in q_model.metrics['fraction_correct']]
            run_episode_success.extend(block_success)
            
            # Store starting arm for each episode
            run_episode_start_arm.extend(q_model.metrics['start_arm'])
            
            # Compute block performance (average across the block)
            block_performance = np.mean(q_model.metrics['fraction_correct'])
            print(f"    Block performance: {block_performance:.2f}")
            
            # Save block-averaged metrics
            for key in block_metrics:
                if key in q_model.metrics:
                    block_metrics[key].append(np.mean(q_model.metrics[key]))
            
            if block_performance >= success_threshold:
                print(f"    Success in block {block_count}")
                consecutive_success += 1
            else:
                print(f"    Failure in block {block_count}")
                consecutive_success = 0
        
        reached_criterion = consecutive_success >= 2
        print(f"  Reached criterion: {reached_criterion}")
        for key in collected_metrics:
            if key == 'blocks_to_criterion':
                collected_metrics[key].append(block_count)
            elif key == 'reached_criterion':
                collected_metrics[key].append(reached_criterion)
            elif key == 'episode_success':
                collected_metrics[key].append(run_episode_success)
            elif key == 'episode_start_arm':
                collected_metrics[key].append(run_episode_start_arm)
            elif key in block_metrics:
                collected_metrics[key].append(block_metrics[key])
    
    avg_metrics = {}
    std_metrics = {}
    for key in collected_metrics:
        if key in ['blocks_to_criterion', 'reached_criterion']:
            avg_metrics[key] = np.mean(collected_metrics[key])
            std_metrics[key] = np.std(collected_metrics[key])
        elif key in ['episode_success', 'episode_start_arm']:
            continue  # Skip these as they're not block-averaged metrics
        else:
            try:
                data = np.array(collected_metrics[key])
                avg_metrics[key] = np.mean(data, axis=0)
                std_metrics[key] = np.std(data, axis=0)
            except ValueError:
                max_len = 0
                for run_data in collected_metrics[key]:
                    if isinstance(run_data, list):
                        max_len = max(max_len, len(run_data))
                    else:
                        max_len = max(max_len, 1)
                
                padded_data = np.full((len(collected_metrics[key]), max_len), np.nan)  # Use actual length instead of num_runs
                for i, run_data in enumerate(collected_metrics[key]):
                    if isinstance(run_data, list):
                        padded_data[i, :len(run_data)] = run_data
                    else:
                        padded_data[i, 0] = run_data
                
                avg_metrics[key] = np.nanmean(padded_data, axis=0)
                std_metrics[key] = np.nanstd(padded_data, axis=0)
    
    return avg_metrics, std_metrics, collected_metrics

def plot_metrics(avg_metrics, std_metrics, collected_metrics):
    # Existing plots (by block)
    if 'fraction_correct' in avg_metrics and len(avg_metrics['fraction_correct']) > 0:
        num_blocks = len(avg_metrics['fraction_correct'])
        blocks = np.arange(1, num_blocks + 1)
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # Plot overall fraction correct and first/second turn fractions
        for key in ['fraction_correct', 'first_turn_fraction', 'second_turn_fraction']:
            if key in avg_metrics and key in std_metrics and len(avg_metrics[key]) > 0:
                axs[0, 0].errorbar(blocks[:len(avg_metrics[key])], avg_metrics[key], 
                                yerr=std_metrics[key],
                                fmt='-o' if key == 'fraction_correct' else 
                                    ('-s' if key == 'first_turn_fraction' else '-^'),
                                capsize=3, 
                                label='Overall' if key == 'fraction_correct' else 
                                    ('1st Turn' if key == 'first_turn_fraction' else '2nd Turn'))
        axs[0, 0].set_title('Fraction Correct (Overall / 1st / 2nd Turns)')
        axs[0, 0].set_xlabel('Block')
        axs[0, 0].set_ylabel('Fraction Correct')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        for i, (row, col, key) in enumerate([
            (0, 1, 'cumulative_reward'),
            (1, 0, 'steps_per_trial'),
            (1, 1, 'epsilon')
        ]):
            if key in avg_metrics and key in std_metrics and len(avg_metrics[key]) > 0:
                axs[row, col].errorbar(blocks[:len(avg_metrics[key])], avg_metrics[key],
                                    yerr=std_metrics[key],
                                    fmt='-o', capsize=3)
        axs[0, 1].set_title('Cumulative Reward per Block')
        axs[0, 1].set_xlabel('Block')
        axs[0, 1].set_ylabel('Cumulative Reward')
        axs[0, 1].grid(True)

        axs[1, 0].set_title('Steps per Trial')
        axs[1, 0].set_xlabel('Block')
        axs[1, 0].set_ylabel('Steps')
        axs[1, 0].grid(True)

        axs[1, 1].set_title('Epsilon Value per Block')
        axs[1, 1].set_xlabel('Block')
        axs[1, 1].set_ylabel('Epsilon')
        axs[1, 1].grid(True)
        
        if 'blocks_to_criterion' in avg_metrics:
            axs[1, 2].axis('off')
            success_rate = avg_metrics.get('reached_criterion', 0) * 100
            axs[1, 2].text(0.5, 0.5, 
                        f'Blocks to Criterion:\nMean: {avg_metrics["blocks_to_criterion"]:.2f}\nStd: {std_metrics["blocks_to_criterion"]:.2f}\n\nSuccess Rate: {success_rate:.1f}%',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    else:
        print("No block-level data to plot")
    
    # --- Updated: Raster Plot for Individual Episodes with Different Colors and No Whitespace ---
    if 'episode_success' in collected_metrics and 'episode_start_arm' in collected_metrics:
        plt.figure(figsize=(10, 6))
        
        # Find the maximum episode count across all runs
        max_episodes = max(len(episodes) for episodes in collected_metrics['episode_success'])
        num_runs = len(collected_metrics['episode_success'])
        
        # Create a 2D array to hold our color-coded data
        # We'll use values: 0 for proximal success (white), 0.5 for failure (blue), 1 for distal success (orange)
        raster_data = np.ones((num_runs, max_episodes)) * 0.5  # Initialize all to failure (blue)
        
        # Fill in the raster data
        for run_idx, (run_episodes, run_start_arms) in enumerate(zip(
                collected_metrics['episode_success'], 
                collected_metrics['episode_start_arm'])):
            
            # Ensure run_episodes and run_start_arms have the same length
            min_len = min(len(run_episodes), len(run_start_arms))
            
            for ep_idx in range(min_len):
                success = run_episodes[ep_idx]
                start_arm = run_start_arms[ep_idx]
                
                if success == 1:
                    # Success - value depends on arm: 0 for proximal (white), 1 for distal (orange)
                    raster_data[run_idx, ep_idx] = 1.0 if start_arm == 1 else 0.0
        
        # Create a custom colormap: white -> blue -> orange
        colors = [(1, 1, 1), (0, 0, 1), (1, 0.5, 0)]  # white, blue, orange
        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # Plot the raster data without any spacing between pixels
        plt.imshow(raster_data, aspect='auto', cmap=cmap, interpolation='none')
        
        # Add labels and title
        plt.xlabel("Episode")
        plt.ylabel("Run")
        
        # Set ticks at block boundaries
        block_size = 32  # From your parameters
        num_blocks = (max_episodes + block_size - 1) // block_size  # Ceiling division
        plt.xticks(np.arange(0, max_episodes, block_size), 
                   np.arange(1, num_blocks + 1))  # Label with block numbers
        
        # Create a custom colorbar legend
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patches as mpatches
        
        # Add a custom legend instead of colorbar
        handles = [
            mpatches.Patch(color='white', label='Success - Proximal Arm'),
            mpatches.Patch(color='blue', label='Failure'),
            mpatches.Patch(color='orange', label='Success - Distal Arm')
        ]
        plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.title("Raster Plot of Episode Success")
        plt.gca().invert_yaxis()  # so that run 1 appears at the top
        plt.tight_layout()
        plt.show()
    else:
        print("No episode-level data to plot")
    
    # Determine how many blocks to plot based on available data
    if 'fraction_correct' in avg_metrics and len(avg_metrics['fraction_correct']) > 0:
        num_blocks = len(avg_metrics['fraction_correct'])
        blocks = np.arange(1, num_blocks + 1)
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # Plot overall fraction of correct trials and overlay first/second turn fractions
        for key in ['fraction_correct', 'first_turn_fraction', 'second_turn_fraction']:
            if key in avg_metrics and key in std_metrics and len(avg_metrics[key]) > 0:
                axs[0, 0].errorbar(blocks[:len(avg_metrics[key])], avg_metrics[key], 
                                yerr=std_metrics[key],
                                fmt='-o' if key == 'fraction_correct' else 
                                    ('-s' if key == 'first_turn_fraction' else '-^'),
                                capsize=3, 
                                label='Overall' if key == 'fraction_correct' else 
                                    ('1st Turn' if key == 'first_turn_fraction' else '2nd Turn'))
        axs[0, 0].set_title('Fraction Correct (Overall / 1st / 2nd Turns)')
        axs[0, 0].set_xlabel('Block')
        axs[0, 0].set_ylabel('Fraction Correct')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        for i, (row, col, key) in enumerate([
            (0, 1, 'cumulative_reward'),
            (1, 0, 'steps_per_trial'),
            (1, 1, 'epsilon')
        ]):
            if key in avg_metrics and key in std_metrics and len(avg_metrics[key]) > 0:
                axs[row, col].errorbar(blocks[:len(avg_metrics[key])], avg_metrics[key],
                                    yerr=std_metrics[key],
                                    fmt='-o', capsize=3)
        axs[0, 1].set_title('Cumulative Reward per Block')
        axs[0, 1].set_xlabel('Block')
        axs[0, 1].set_ylabel('Cumulative Reward')
        axs[0, 1].grid(True)

        axs[1, 0].set_title('Steps per Trial')
        axs[1, 0].set_xlabel('Block')
        axs[1, 0].set_ylabel('Steps')
        axs[1, 0].grid(True)

        axs[1, 1].set_title('Epsilon Value per Block')
        axs[1, 1].set_xlabel('Block')
        axs[1, 1].set_ylabel('Epsilon')
        axs[1, 1].grid(True)
        
        # Add a text box showing blocks to criterion
        if 'blocks_to_criterion' in avg_metrics:
            axs[1, 2].axis('off')
            success_rate = avg_metrics.get('reached_criterion', 0) * 100
            axs[1, 2].text(0.5, 0.5, 
                        f'Blocks to Criterion:\nMean: {avg_metrics["blocks_to_criterion"]:.2f}\nStd: {std_metrics["blocks_to_criterion"]:.2f}\n\nSuccess Rate: {success_rate:.1f}%',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot")

if __name__ == "__main__":
    NUM_RUNS = 20
    BLOCK_SIZE = 32
    SUCCESS_THRESHOLD = 0.7
    MAX_BLOCKS = 32
    
    avg_metrics, std_metrics, collected_metrics = run_multiple_trials(
        num_runs=NUM_RUNS, 
        block_size=BLOCK_SIZE, 
        success_threshold=SUCCESS_THRESHOLD,
        max_blocks=MAX_BLOCKS
    )
    
    # Print out success statistics
    success_count = sum(collected_metrics.get('reached_criterion', []))
    total_runs = len(collected_metrics.get('reached_criterion', []))
    success_rate = (success_count / total_runs * 100) if total_runs > 0 else 0
    
    print(f"Average blocks to criterion: {avg_metrics.get('blocks_to_criterion', 'N/A')}")
    print(f"Success rate: {success_count}/{total_runs} ({success_rate:.1f}%)")
    # Pass all required parameters to plot_metrics
    plot_metrics(avg_metrics, std_metrics, collected_metrics)