import pandas as pd
from thirtytwostatepygame import Environment, QLearningModel, Screen
from modules.mapping import STATES_TO_COORDINATES
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load and filter rat data from CSV
    rat_data = pd.read_csv("main_project/modules/rat1_data.csv")
    final_session = input("Enter the final session number: ")
    rat_data = rat_data[(rat_data["Session"] >= 1) & (rat_data["Session"] <= int(final_session))]
    
    q_model_percentage_correct_arr = []
    rat_data_percentage_correct_arr = []


    
    # Initialize environment and model
    screen = Screen()
    maze_width = screen.width//screen.cell_size
    maze_height = (screen.height-50)//screen.cell_size
    environment = Environment(maze_width, maze_height, rat_data.iloc[0]["State"], start=[0,2])
    q_model = QLearningModel()

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
        previous_trial = int(abs(previous_trial))

        # Reset on new trial or session
        if (current_trial != previous_trial) or (previous_session != session):
            print(f"Resetting environment for Session {session}, Trial {trial}")
            correctness_index = 0
            environment.full_reset(start_state=state)
            # Ensure goal_seeking is set correctly at start of trial
            environment.goal_seeking = True if trial > 0 else False
            force_state = True
        
        # Force environment state to match data when needed
        if force_state or environment.state != state:
            # Directly set state values to match the data
            environment.state = state
            environment.x = STATES_TO_COORDINATES[state-1][0]
            environment.y = STATES_TO_COORDINATES[state-1][1]
            environment._infer_orientation()
            
            # Only log if it wasn't a forced update
            if not force_state and environment.state != state:
                print(f"Fixing state mismatch: env={environment.state} -> data={state}, index={index}")
            force_state = False
        
        # Get model prediction
        prediction = q_model.choose_action(environment.state-1)
        
        # Determine expected actions based on start position
        expected_first = 1 if start == 2 else 2  # left or right
        expected_second = 1
        
        # Track correctness
        if correctness_index == 1:
            q_model_first_correct = (prediction == expected_first)
            rat_first_correct = (action == expected_first)
        elif correctness_index == 2:
            q_model_second_correct = (prediction == expected_second)
            rat_second_correct = (action == expected_second)
            if q_model_first_correct and q_model_second_correct:
                q_model_correct += 1
            if rat_first_correct and rat_second_correct:
                rat_data_correct += 1
            q_model_total += 1
            rat_data_total += 1
            
        q_model_percentage_correct = (q_model_correct / q_model_total) if q_model_total > 0 else 0
        rat_data_percentage_correct = (rat_data_correct / rat_data_total) if rat_data_total > 0 else 0
        q_model_percentage_correct_arr.append(q_model_percentage_correct)
        rat_data_percentage_correct_arr.append(rat_data_percentage_correct)
        
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
        previous_trial = trial
        previous_session = session
        correctness_index += 1
        
        # Take step using the rat's action
        q_model.step(environment, action, start=next_start)


    # Print results
    print(f"Model performance: {q_model_correct}/{q_model_total} = {q_model_correct/q_model_total:.2%} correct")
    print(f"Rat performance: {rat_data_correct}/{rat_data_total} = {rat_data_correct/rat_data_total:.2%} correct")

    x_axis = np.arange(len(rat_data)-1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, rat_data_percentage_correct_arr, marker="o", linestyle="-", color = "blue")
    plt.plot(x_axis, q_model_percentage_correct_arr, marker="o", linestyle="-", color = "red")
    plt.xlabel("Trial Index")
    plt.ylabel("Percentage Correct")
    plt.title("Rat Performance Over Trials")
    plt.xticks(x_axis, [f"Trial {i}" for i in range(len(rat_data_percentage_correct_arr))])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()