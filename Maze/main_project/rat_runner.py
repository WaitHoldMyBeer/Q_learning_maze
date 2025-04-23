import pandas as pd
from thirtytwostatepygame import Environment, QLearningModel, Screen
from modules.mapping import STATES_TO_COORDINATES
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import math
import random
import optuna

rat_dfs =[]
for i in range(1,13):
    df = pd.read_csv(f"modules/00{i}.csv")
    df['RatID'] = i
    rat_dfs.append(df)
all_data = pd.concat(rat_dfs, ignore_index = True)

def objective(trial):
    # Suggest values for each hyperparameter
    alpha = trial.suggest_float('alpha', 0.01, 1.0, log=True)
    gamma = trial.suggest_float('gamma', 0.5, 0.99)
    epsilon = trial.suggest_float('epsilon', 0.0, 1.0)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.90, 0.9999)
    min_epsilon = trial.suggest_float('min_epsilon', 0.0, 0.5)
    reward_size = trial.suggest_int('reward_size', 0, 50)
    policy = trial.suggest_categorical('policy', ['epsilon_greedy', 'softmax'])
    temperature = trial.suggest_float('temperature', 0.1, 5.0)
    temperature = trial.suggest_float('temperature_decay')
    use_softmax = (policy == 'softmax')
    
    # Evaluate this parameter set on the training rats
    agreements = []
    train_rats = [1,2,4,5,6,7,8,9,10,11]
    for rat_id in train_rats:             # train_rats = [1,...,10]
        rat_data = all_data[all_data.RatID == rat_id]
        score = run_training_on_rat(rat_data,  # performs the simulation loop
                                    alpha, gamma, epsilon, 
                                    epsilon_decay, min_epsilon, 
                                    reward_size, use_softmax, temperature)
        agreements.append(score)
    mean_agreement = sum(agreements) / len(agreements)
    return mean_agreement   # or -mean_agreement if minimizing

def shannon_entropy_bool_pairs(samples):
    """
    Calculate Shannon entropy (in bits) for a list of samples,
    each represented as a tuple (e.g., (0,1) or (True, False)).
    """
    N = len(samples)
    counts = Counter(samples)
    entropy = 0.0
    for count in counts.values():
        p = count / N
        entropy -= p * math.log2(p)
    return entropy

def sliding_window_entropy(mat, window_size=8):
    """
    Compute Shannon entropy over sliding windows of length `window_size`
    on a 2×N numpy array. Returns a 1D numpy array of length N − window_size + 1.
    """
    if mat.ndim != 2 or mat.shape[0] != 2:
        raise ValueError("Input matrix must have shape (2, N)")
   
    N = mat.shape[1]
    num_windows = N - window_size + 1
    entropies = np.zeros(num_windows)
   
    for i in range(num_windows):
        segment = mat[:, i:i+window_size]
        # Build the list of 8 samples as (row0[i], row1[i]) tuples
        samples = list(zip(segment[0], segment[1]))
        entropies[i] = shannon_entropy_bool_pairs(samples)
   
    return entropies


def main():
    total = 0
    for i in range(10):
        np.random.seed(40+i)
        random.seed(40+i)
        total+=q_model()
    #print("Average Mean squared loss =", total/10)


def run_training_on_rat(rat_data, alpha, gamma, epsilon, epsilon_decay, min_epsilon, reward_size, use_softmax, temperature):
    #rat_data = rat_data[(rat_data["Session"] >= 1) & (rat_data["Session"] <= int(final_session))]
    rat_data = rat_data[(((rat_data["Session"] >= 1) & (rat_data["Session"] <= 8)) | 
                         #(rat_data["Session"] == 12)
                        ((rat_data["Session"] == 12))
                        )]

    #np.convolve(ndarray, np.ones(size)/size, mode = "full")


    model_actions = []
    rat_actions = []
    # Initialize environment and model
    screen = Screen()
    maze_width = screen.width//screen.cell_size
    maze_height = (screen.height-50)//screen.cell_size
    environment = Environment(maze_width, maze_height, int(rat_data.iloc[0]["State"]), start=[0,2])
    q_model = QLearningModel(alpha = alpha, gamma = gamma, epsilon = epsilon, epsilon_decay=epsilon_decay, min_epsilon = min_epsilon, reward_size = reward_size, soft_max=use_softmax, temperature = temperature)

    # Tracking variables
    previous_session = 1
    correctness_index = 0
    
    # Measurement variables

    # Track when to force state
    force_state = True
    previous_trial = 1

    # Process all data
    for index in range(len(rat_data)-1):

        session = int(rat_data.iloc[index]['Session'])
        state = int(rat_data.iloc[index]['State'])
        action = int(rat_data.iloc[index]['Action'])
        trial = int(rat_data.iloc[index]['Trial'])
        start = rat_data.iloc[index]['Start']
        
        current_trial = int(abs(trial))

        q_model_correct = 0
        rat_data_correct = 0

        # Reset on new trial or session
        if (current_trial != previous_trial) or (previous_session != session):
            #print("Trial =", trial)
            correctness_index = 0
            environment.full_reset(start_state=state)
            # Ensure goal_seeking is set correctly at start of trial
            environment.goal_seeking = True if trial > 0 else False
            force_state = True
        

        # Force environment state to match data when needed
        if (force_state
            or environment.state != state
            ):
            # Directly set state values to match the data
            environment.state = state
            environment.x = STATES_TO_COORDINATES[state-1][0]
            environment.y = STATES_TO_COORDINATES[state-1][1]
            environment._infer_orientation()
            
            # Only log if it wasn't a forced update
            if not force_state and environment.state != state:
                print(f"Fixing state mismatch: env={environment.state} -> data={state}, index={index}")
            force_state = False
        
        

        #get model prediction
        prediction = q_model.choose_action(environment.state-1)
        model_actions.append(prediction)
        rat_actions.append(action)
        #print(f"CSV state = {state}. Environment state = {environment.state}. Rat action = {action}. Q Model action = {prediction}")

            

        # Take step based on rat's actual action
        next_start = rat_data.iloc[index + 1]['Start'] if index + 1 < len(rat_data) else None
        next_trial = rat_data.iloc[index + 1]['Trial'] if index + 1 < len(rat_data) else None
        
        # Check if we need to handle a goal/return phase transition
        if index + 1 < len(rat_data):
            next_state = rat_data.iloc[index + 1]['State']
            # Force state on goal/return transitions (negative trial indicates return phase)
            if (trial > 0 and next_trial < 0) or (trial < 0 and next_trial > 0):
                force_state = True
                
        # Update tracking variables
        previous_trial = current_trial
        previous_session = session
        correctness_index += 1
        
        # Take step using the rat's action
        #
        q_model.step(environment, action = action, start=next_start)
    return compute_agreement(rat_actions=rat_actions, model_actions=model_actions)

def compute_agreement(rat_actions, model_actions):
    """Compute proportion of actions where model's choice matches the rat's choice."""
    assert len(rat_actions) == len(model_actions)
    total_steps = len(rat_actions)
    matching = sum(1 for r, m in zip(rat_actions, model_actions) if r == m)
    return matching / total_steps


def q_model():
    
    # Load and filter rat data from CSV
    rat_data = pd.read_csv("modules/003.csv")
    rat_data_session_12 = rat_data[(rat_data["Session"] == 12) & (abs(rat_data["Trial"]) > 16)]
    #final_session = 8
    #rat_data = rat_data[(rat_data["Session"] >= 1) & (rat_data["Session"] <= int(final_session))]
    rat_data = rat_data[(((rat_data["Session"] >= 1) & (rat_data["Session"] <= 8)) | 
                         #(rat_data["Session"] == 12)
                        ((rat_data["Session"] == 12) & (abs(rat_data["Trial"]) <= 16))
                        )]

    #np.convolve(ndarray, np.ones(size)/size, mode = "full")


    q_model_correct_arr = []
    rat_data_correct_arr = []
    rat_first_turn = []
    rat_second_turn = []
    q_model_first_turn = []
    q_model_second_turn = []
    q_model_first_turn_values_north = []
    q_model_first_turn_values_south = []
    q_model_second_turn_values_left = []
    q_model_second_turn_values_right = []

    reward_size = 300
    # Initialize environment and model
    screen = Screen()
    maze_width = screen.width//screen.cell_size
    maze_height = (screen.height-50)//screen.cell_size
    environment = Environment(maze_width, maze_height, rat_data.iloc[0]["State"], start=[0,2])
    q_model = QLearningModel(alpha = .661, gamma = .988, epsilon = .1275, epsilon_decay=.9977, min_epsilon=.116, reward_size = 50, soft_max=True, temperature = 2.110)

    # Tracking variables
    previous_session = 1
    correctness_index = 0
    
    # Measurement variables
    q_model_total = 0
    q_model_correct = 0
    rat_data_total = 0
    rat_data_correct = 0

    # Track when to force state
    force_state = True
    previous_trial = 1

    # Process all data
    for index in range(len(rat_data)-1):

        session = rat_data.iloc[index]['Session']
        state = rat_data.iloc[index]['State']
        action = rat_data.iloc[index]['Action']
        trial = rat_data.iloc[index]['Trial']
        start = rat_data.iloc[index]['Start']
        
        current_trial = int(abs(trial))

        q_model_correct = 0
        rat_data_correct = 0

        # Reset on new trial or session
        if (current_trial != previous_trial) or (previous_session != session):
            #print("Trial =", trial)
            correctness_index = 0
            environment.full_reset(start_state=state)
            # Ensure goal_seeking is set correctly at start of trial
            environment.goal_seeking = True if trial > 0 else False
            force_state = True
        

        # Force environment state to match data when needed
        if (force_state
            or environment.state != state
            ):
            # Directly set state values to match the data
            environment.state = state
            environment.x = STATES_TO_COORDINATES[state-1][0]
            environment.y = STATES_TO_COORDINATES[state-1][1]
            environment._infer_orientation()
            
            # Only log if it wasn't a forced update
            if not force_state and environment.state != state:
                print(f"Fixing state mismatch: env={environment.state} -> data={state}, index={index}")
            force_state = False
        
        

        #get model prediction
        prediction = q_model.choose_action(environment.state-1)
        

        #print(f"CSV state = {state}. Environment state = {environment.state}. Rat action = {action}. Q Model action = {prediction}")

        # Determine expected actions based on start position
        expected_first = 1 if start == 2 else 2  # left or right
        expected_second = 1
        
        # Track correctness
        if correctness_index == 1:
            q_model_first_correct = (prediction == expected_first)
            rat_first_correct = (action == expected_first)
            rat_first_turn.append(action-2)
            q_model_first_turn.append(prediction == 1)
            q_model_first_turn_values_north.append(q_model.q_table[1-1,:].copy())
            q_model_first_turn_values_south.append(q_model.q_table[3-1,:].copy())

        elif correctness_index == 2:
            if state == 6:
                expected_second = 3
            q_model_second_correct = (prediction == expected_second)
            rat_second_correct = (action == expected_second)
            if q_model_first_correct and q_model_second_correct:
                q_model_correct = 1
            if rat_first_correct and rat_second_correct:
                rat_data_correct = 1
            q_model_total += 1
            rat_data_total += 1
            q_model_correct_arr.append(q_model_correct)
            rat_data_correct_arr.append(rat_data_correct)
            rat_second_turn.append(state-2)
            q_model_second_turn.append(prediction == 1)
            q_model_second_turn_values_left.append(q_model.q_table[6-1,:].copy())
            q_model_second_turn_values_right.append(q_model.q_table[8-1,:].copy())

        # Take step based on rat's actual action
        next_start = rat_data.iloc[index + 1]['Start'] if index + 1 < len(rat_data) else None
        next_trial = rat_data.iloc[index + 1]['Trial'] if index + 1 < len(rat_data) else None
        
        # Check if we need to handle a goal/return phase transition
        if index + 1 < len(rat_data):
            next_state = rat_data.iloc[index + 1]['State']
            # Force state on goal/return transitions (negative trial indicates return phase)
            if (trial > 0 and next_trial < 0) or (trial < 0 and next_trial > 0):
                force_state = True
                
        # Update tracking variables
        previous_trial = current_trial
        previous_session = session
        correctness_index += 1
        
        # Take step using the rat's action
        #
        q_model.step(environment, action = action, start=next_start)

    acquisition_trials_num = len(q_model_correct_arr)

    # session 12 data
    session_12_indexer = 0
    q_model_trial = 0
    environment = Environment(maze_width, maze_height, rat_data.iloc[0]["State"], goal_corner = 3, start=[0,2])
    total_reward = 0
    while session_12_indexer < len(rat_data_session_12)-1:
        
        session = rat_data_session_12.iloc[session_12_indexer]['Session']
        state = rat_data_session_12.iloc[session_12_indexer]['State']
        action = rat_data_session_12.iloc[session_12_indexer]['Action']
        trial = rat_data_session_12.iloc[session_12_indexer]['Trial']
        start = rat_data_session_12.iloc[session_12_indexer]['Start']
        
        current_trial = int(abs(trial))
        previous_trial = int(abs(previous_trial))
        
        
        expected_first = 1 if start == 2 else 2  # left or right
        expected_second = 1

        if (current_trial != previous_trial):
            environment.full_reset(start_state=rat_data_session_12.iloc[session_12_indexer]['State'])
            correctness_index = 0
            environment.goal_seeking = True
            #print("Q Model Trial")

            # q_model trial runs
            while True:
                q_action, reward = q_model.step(environment, start = start)
                total_reward += reward
                #print("Q action =", q_action, "Reward = ", reward, "Cumulative Reward =", total_reward, "Start =", start, "State =", environment.state, "correctness_index =", correctness_index, "expected first?:", (q_action == expected_first))
                if correctness_index == 1:
                    q_model_first_correct = (q_action == expected_first)
                    q_model_first_turn.append(q_action == 1)
                    q_model_first_turn_values_north.append(q_model.q_table[1-1, :].copy())
                    q_model_first_turn_values_south.append(q_model.q_table[3-1,:].copy())
                if correctness_index == 2:
                    q_model_second_correct = (q_action == expected_second)
                    q_model_correct = 0
                    if q_model_first_correct and q_model_second_correct:
                        q_model_correct = 1
                    q_model_correct_arr.append(q_model_correct)
                    q_model_second_turn.append(q_action == 1)
                    q_model_second_turn_values_left.append(q_model.q_table[6-1,:].copy())
                    q_model_second_turn_values_right.append(q_model.q_table[8-1,:].copy())
                if (environment.state == environment.goal_corner_state):
                    break
                correctness_index += 1
            correctness_index = 0
            q_model_trial+=1

        if correctness_index == 1:
            rat_first_correct = (action == expected_first)
            rat_first_turn.append(action-2)
        elif correctness_index == 2:
            rat_second_correct = (action == expected_second)
            rat_second_turn.append(action-2)
            rat_data_correct = 0
            if rat_first_correct and rat_second_correct:
                rat_data_correct = 1
            rat_data_correct_arr.append(rat_data_correct)
        previous_trial = trial
        correctness_index += 1
        session_12_indexer += 1


    q_model_correct_arr = np.array(q_model_correct_arr)
    rat_data_correct_arr = np.array(rat_data_correct_arr)
    q_model_first_second_turn = np.vstack([q_model_first_turn, q_model_second_turn])
    rat_first_second_turn = np.vstack([rat_first_turn, rat_second_turn])
    q_model_first_turn_values_north = np.array(q_model_first_turn_values_north)
    q_model_first_turn_values_south = np.array(q_model_first_turn_values_south)
    q_model_second_turn_values_left = np.array(q_model_second_turn_values_left)
    q_model_second_turn_values_right = np.array(q_model_second_turn_values_right)


    q_model_entropy = sliding_window_entropy(q_model_first_second_turn)
    rat_data_entropy = sliding_window_entropy(rat_first_second_turn)


    window = 8
    og_convolver = np.ones(window)/window
    convolver_len = 2*window-1
    convolver = np.zeros(2*window-1)
    convolver[:window] = 15/8
    rat_rolling_average = np.convolve(rat_data_correct_arr, convolver/convolver_len, mode="full")
    q_model_rolling_average = np.convolve(q_model_correct_arr, convolver/convolver_len, mode="full")
    
    total_error = 0
    for i in range(len(rat_rolling_average)):
        total_error += (rat_rolling_average[i]-q_model_rolling_average[i])**2

    #return total_error/len(rat_rolling_average)

    rolling_avg_len = len(rat_rolling_average)
    q_values_len = len(q_model_first_turn_values_north)
    entropy_len = len(q_model_entropy)

    # Create separate x-axes for different plot types
    rolling_x = np.arange(rolling_avg_len)
    q_values_x = np.arange(q_values_len)
    entropy_x = np.arange(entropy_len)
    q_values_height = reward_size
    q_values_floor = -3
    def display_figure():
        plt.figure(figsize=(12, 8))
        
        # First subplot - Rolling Averages
        plt.subplot(231)
        plt.axis((0, rolling_avg_len, 0, 1))
        plt.plot(rolling_x, rat_rolling_average, marker="o", linestyle="-", color="blue", label="Rat Rolling Average")
        plt.plot(rolling_x, q_model_rolling_average, marker="o", linestyle="-", color="red", label="Q Model's Rolling Average")
        plt.xlabel("Trial Index")
        plt.axvline(x=acquisition_trials_num, color="green", linestyle="--", label="Session 12")
        plt.ylabel("Percentage Correct")
        plt.title("Rat Performance Over Trials")
        plt.legend()

        # Second subplot - Entropy Values
        plt.subplot(232)
        plt.plot(entropy_x, q_model_entropy, "-", color="red", label="Q model Entropy")
        plt.plot(entropy_x, rat_data_entropy, "-", color="blue", label="Rat Data Entropy")
        plt.axis((0, entropy_len, 0, 2))
        plt.xlabel("Trial Index")
        plt.ylabel("Entropy Value")
        plt.title("Entropy over Trials")
        plt.legend()
        
        # Third subplot - North Q Values
        plt.subplot(233)
        plt.plot(q_values_x, q_model_first_turn_values_north[:,0], marker="o", linestyle="-", color="red", label="Forward")
        plt.plot(q_values_x, q_model_first_turn_values_north[:,1], marker="o", linestyle="-", color="blue", label="Left")
        plt.plot(q_values_x, q_model_first_turn_values_north[:,2], marker="o", linestyle="-", color="green", label="Right")
        plt.plot(q_values_x, q_model_first_turn_values_north[:,3], marker="o", linestyle="-", color="purple", label="Backward")
        plt.axis((0, q_values_len, q_values_floor, q_values_height))
        plt.xlabel("Trial Index")
        plt.ylabel("Q Value")
        plt.title("Q Model First Turn Values (North)")
        plt.legend()

        # Fourth subplot - South Q Values  
        plt.subplot(234)
        plt.plot(q_values_x, q_model_first_turn_values_south[:,0], marker="o", linestyle="-", color="red", label="Forward")
        plt.plot(q_values_x, q_model_first_turn_values_south[:,1], marker="o", linestyle="-", color="blue", label="Left")
        plt.plot(q_values_x, q_model_first_turn_values_south[:,2], marker="o", linestyle="-", color="green", label="Right")
        plt.plot(q_values_x, q_model_first_turn_values_south[:,3], marker="o", linestyle="-", color="purple", label="Backward")
        plt.axis((0, q_values_len, q_values_floor, q_values_height))
        plt.xlabel("Trial Index")
        plt.ylabel("Q Value")
        plt.title("Q Model First Turn Values (South)")
        plt.legend()

        # Fifth subplot - Left Q Values
        plt.subplot(235)
        plt.plot(q_values_x, q_model_second_turn_values_left[:,0], marker="o", linestyle="-", color="red", label="Forward")
        plt.plot(q_values_x, q_model_second_turn_values_left[:,1], marker="o", linestyle="-", color="blue", label="Left")
        plt.plot(q_values_x, q_model_second_turn_values_left[:,2], marker="o", linestyle="-", color="green", label="Right")
        plt.plot(q_values_x, q_model_second_turn_values_left[:,3], marker="o", linestyle="-", color="purple", label="Backward")
        plt.axis((0, q_values_len, q_values_floor, q_values_height))
        plt.xlabel("Trial Index")
        plt.ylabel("Q Value")
        plt.title("Q Model Second Turn Values (Left)")
        plt.legend()
        
        # Sixth subplot - Right Q Values
        plt.subplot(236)
        plt.plot(q_values_x, q_model_second_turn_values_right[:,0], marker="o", linestyle="-", color="red", label="Forward")
        plt.plot(q_values_x, q_model_second_turn_values_right[:,1], marker="o", linestyle="-", color="blue", label="Left")
        plt.plot(q_values_x, q_model_second_turn_values_right[:,2], marker="o", linestyle="-", color="green", label="Right")
        plt.plot(q_values_x, q_model_second_turn_values_right[:,3], marker="o", linestyle="-", color="purple", label="Backward")
        plt.axis((0, q_values_len, q_values_floor, q_values_height))
        plt.xlabel("Trial Index")
        plt.ylabel("Q Value")
        plt.title("Q Model Second Turn Values (Right)")
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle("Q Model vs Rat Data Analysis", fontsize=16)
        plt.subplots_adjust(top=0.88)
        plt.grid(True)
        plt.show()
    
    display_figure()

def optuna_func():
    study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="50 Q Value Optimizer", direction="maximize")
    study.optimize(objective, n_trials=50)
     # Print the best parameters and score
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()

if __name__ == "__main__":
    q_model()