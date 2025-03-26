import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import torch  # Import torch
import copy


np.random.seed(42)
torch.manual_seed(42)
import random
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# region --- Constants ---

STATE_VECTORS = [(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0),
                 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1),
                 ]


CORRECT_ACQ = [(1,0,0,0), (0,0,0,1), (0,1,0,0), (0,0,0,1), 
               (0,0,1,0), (0,0,0,1), (0,0,0,1), (1,0,0,0), 
               (0,0,0,1), (1,0,0,0), (0,0,0,1), (1,0,0,0), 
               (0,0,0,1), (0,1,0,0), (0,0,0,1), (1,0,0,0), 
               (1,0,0,0), (0,0,0,1), (0,1,0,0), (0,0,0,1), 
               (0,0,0,1), (0,1,0,0), (0,0,1,0), (0,0,0,1), 
               (0,0,0,1), (0,0,0,0), (0,1,0,0), (0,0,0,1), 
                (0,1,0,0), (0,0,0,1), (0,1,0,0), (0,0,0,1)] #(L,R)

CORRECT_EXT = [(0,1,0,0), (0,0,0,1), (1,0,0,0), (0,0,0,1), 
               (0,0,0,1), (0,0,1,0), (0,0,1,0), (0,0,0,1), 
               (0,0,0,1), (1,0,0,0), (0,0,0,1), (0,1,0,0), 
               (0,0,0,1), (1,0,0,0), (0,0,0,1), (1,0,0,0), 
               (0,0,1,0), (0,0,0,1), (0,0,0,1), (0,1,0,0), 
               (1,0,0,0), (0,0,0,1), (1,0,0,0), (0,0,0,1), 
               (1,0,0,0), (0,0,0,1), (0,1,0,0), (0,0,0,1), 
                (0,0,0,1), (0,0,0,0), (0,1,0,0), (0,0,0,1)] #(L,R)

CORRECT_CHOICE_ACQ_ABSOLUTE = [0, 3, 1, 3,
                  2, 3, 3, 0, 
                  3, 0, 3, 0,
                  3, 1, 3, 0,
                  0, 3, 1, 3,
                  3, 1, 2, 3,
                  3, -1, 1, 3,
                  1, 3, 1, 3]  #(scores second choice absolutely)

CORRECT_CHOICE_EXT_ABSOLUTE = [1, 3, 0, 3,
               3, 2, 2, 3, 
               3, 0, 3, 1, 
               3, 0, 3, 0, 
               2, 3, 3, 1, 
               0, 3, 0, 3, 
               0, 3, 1, 3, 
               3, -1, 1, 3]  #(scores second choice absolutely)

CORRECT_CHOICE_ACQ = [0, 3, 1, 3,
                  2, 0, 3, 0, 
                  3, 0, 3, 0,
                  3, 1, 3, 0,
                  0, 3, 1, 3,
                  3, 1, 2, 3,
                  3, -1, 1, 3,
                  1, 3, 1, 3]  #(scores second choice assuming first was correct)

CORRECT_CHOICE_EXT = [1, 3, 0, 3,
               3, 2, 2, 0, 
               3, 0, 3, 1, 
               3, 0, 3, 0, 
               2, 3, 3, 1, 
               0, 3, 0, 3, 
               0, 3, 1, 3, 
               3, -1, 1, 3]  #(scores second choice assuming first was correct)

TRANSITIONS_OPEN = [(8,6,4,2), (10,12,2,1), (6,8,2,4), (16,14,4,3), 
               (2,4,8,6), (20,18,6,5), (4,2,6,8), (22,24,8,7), 
               (9,1,12,10), (17,32,10,9), (1,11,10,12), (26,21,12,11),
               (3,13,16,14), (30,19,14,13), (15,3,14,16), (23,28,16,15),
               (5,17,20,18), (32,9,18,17), (19,5,18,20), (13,30,20,19),
               (21,7,24,22), (11,26,22,21), (7,23,22,24), (28,15,24,23),
               (21,11,25,26), (25,25,26,25), (15,23,27,28), (27,27,28,27),
               (19,13,29,30), (29,29,30,29), (9,17,31,32), (32,32,32,31) ] #(L,R)

TRANSITIONS_S1 = [(8,6,1,2), (1,1,2,1), (3,3,3,4), (4,4,4,3), 
               (2,5,8,6), (20,18,6,5), (7,2,6,8), (22,24,8,7), 
               (10,10,9,10), (17,32,10,9), (12,12,11,12), (26,21,12,11),
               (14,14,13,14), (30,19,14,13), (16,16,15,16), (23,28,16,15),
               (5,17,20,18), (32,9,18,17), (19,5,18,20), (13,30,20,19),
               (21,7,24,22), (11,26,22,21), (7,23,22,24), (28,15,24,23),
               (21,11,25,26), (25,25,26,25), (15,23,27,28), (27,27,28,27),
               (19,13,29,30), (29,29,30,29), (9,17,31,32), (32,32,32,31) ] #(L,R)

TRANSITIONS_S3 = [(1,1,1,2), (2,2,2,1), (6,8,3,4), (3,3,4,3), 
               (5,4,8,6), (20,18,6,5), (4,7,6,8), (22,24,8,7), 
               (10,10,9,10), (17,32,11,9), (12,12,11,12), (26,21,12,11),
               (14,14,13,14), (30,19,14,13), (16,16,15,16), (23,28,16,15),
               (5,17,20,18), (32,9,18,17), (19,5,18,20), (13,30,20,19),
               (21,7,24,22), (11,26,22,21), (7,23,22,24), (28,15,24,23),
               (21,11,25,26), (25,25,26,25), (15,23,27,28), (27,27,28,27),
               (19,13,29,30), (29,29,30,29), (9,17,31,32), (32,32,32,31) ] #(L,R)

TRANSITIONS_OPEN = [
    tuple(value - 1 for value in transitions)
    for transitions in TRANSITIONS_OPEN
]
TRANSITIONS_S1 = [
    tuple(value - 1 for value in transitions)
    for transitions in TRANSITIONS_S1
]
TRANSITIONS_S3 = [
    tuple(value - 1 for value in transitions)
    for transitions in TRANSITIONS_S3
]

#endregion

# Check if MPS is available and set the device
use_gpu = 0
if use_gpu:
    device = torch.device("mps")
    print("MPS is available and set as the device.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU instead.")

class ChoiceLoggingCallback(BaseCallback):
    def __init__(self, env, bin_size=10, verbose=0):
        super(ChoiceLoggingCallback, self).__init__(verbose)
        self.env = env
        self.bin_size = bin_size
        self.log = {
            "first_choice_correct": [],
            "second_choice_correct": [],
            "both_choices_correct": [],
            "softmax_probs_s1": [], #north start
            "softmax_probs_s2": [], #north choice 2 (alt task after correct choice 1)
            "softmax_probs_s3": [], #south start
            "softmax_probs_s4": [], #north choice 2 (alt task after incorrect choice 1)
            "softmax_probs_s5": [], #west start (alt task choice 1)
            "softmax_probs_s6": [], #west choice 2 (after incorrect choice 1)
            "softmax_probs_s8": [], #east choice 2 (after correct choice 1)
            "critic_values": {i: [] for i in range(len(STATE_VECTORS))},  # Critic values for each state
        }
        self.logged_choice1 = False  # Store the previous observation
        self.logged_choice2 = False  # Store the previous observation
        self.logged_state_values = False  
        self.prev_obs = None  # Store the previous observation
        self.prev_action = None  # Store the previous action
        self.extinction = False
        self.state_list = []

    def _on_training_start(self) -> None:

        self.prev_obs = self.training_env.get_attr("state")[0]

    def _on_step(self) -> bool:
        
        # Access the environment's state
        obs = self.training_env.get_attr("state")[0]
        self.prev_action = self.locals["actions"]
        reporting = True
        reporting = False

        # Only log if there's a previous observation
        if self.prev_obs is not None:

            snum = next((i for i, x in enumerate(self.prev_obs) if x != 0), None)

            #if self.extinction:
            #    chose_correctly = self.prev_action==CORRECT_CHOICE_EXT[snum]
            #else:
            chose_correctly = self.prev_action==CORRECT_CHOICE_ACQ[snum]
            
            self.state_list.append(snum)
            if snum+1 in (26, 28, 30, 32):  # and any(x in self.state_list for x in (4, 1, 6, 3)):
                #print(self.state_list)
                self.state_list = []
                self.logged_choice1 = False  
                self.logged_choice2 = False  
                self.logged_state_values = False  

                # print(self.num_timesteps)
                # print(obs)
                # print(len(self.log["first_choice_correct"]))

            # Convert the observation to a tensor and move it to the model's device
            obs_tensor = torch.tensor(self.prev_obs, dtype=torch.float32).unsqueeze(0).to(self.model.device)

            # Compute action probabilities using the policy's distribution
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(obs_tensor)
                action_probs = distribution.distribution.probs.cpu().numpy()  # Extract probabilities and move to CPU

            # Determine correctness for first choice
            version = self.env.version

            logged_a_choice = False

            #---- first choice cases
            if (not self.logged_choice1) and self.prev_action in (0, 1):

                if snum+1 == 1:   # Choosing L or R for the first time this trial in s1
                    self.log["softmax_probs_s1"].append(action_probs[0][0])  # Log softmax for s1
                    self.log["softmax_probs_s3"].append(np.nan)  # Append NaN for s3
                    self.log["softmax_probs_s5"].append(np.nan)  # Append NaN for s5
                    self.log["first_choice_correct"].append(float(chose_correctly))
                    self.logged_choice1 = True
                    # if reporting:
                    #     print(f"TS:{self.num_timesteps} State:s1 Action:{self.prev_action} len1:{len(self.log['first_choice_correct'])} len2:{len(self.log['second_choice_correct'])}")
                    
                elif snum+1 == 3:  # Choosing L or R for the first time this trial in s3
                    self.log["softmax_probs_s3"].append(action_probs[0][0])  # Log softmax for s3
                    self.log["softmax_probs_s1"].append(np.nan)  # Append NaN for s1
                    self.log["softmax_probs_s5"].append(np.nan)  # Append NaN for s5
                    self.log["first_choice_correct"].append(float(chose_correctly))
                    self.logged_choice1 = True
                    # if reporting:
                    #     print(f"TS:{self.num_timesteps} State:s3 Action:{self.prev_action} len1:{len(self.log['first_choice_correct'])} len2:{len(self.log['second_choice_correct'])}")

                elif snum+1 == 5:   # Choosing L or R for the first time this trial in s5
                    self.log["softmax_probs_s5"].append(action_probs[0][0])  # Log softmax for s5
                    self.log["softmax_probs_s1"].append(np.nan)  # Append NaN for s1
                    self.log["softmax_probs_s3"].append(np.nan)  # Append NaN for s3
                    self.log["first_choice_correct"].append(float(chose_correctly))
                    self.logged_choice1 = True
                    # if reporting:
                    #     print(f"TS:{self.num_timesteps} State:s4 Action:{self.prev_action} len1:{len(self.log['first_choice_correct'])} len2:{len(self.log['second_choice_correct'])}")

            #---- second choice cases
            if (not self.logged_choice2) and self.prev_action in (0, 1):

                if snum+1 == 8:  # # Last choice made in s8 (first choice was correct)
                    self.log["softmax_probs_s8"].append(action_probs[0][0])  # Log softmax for s8
                    self.log["softmax_probs_s6"].append(np.nan)  # Append NaN for s6
                    self.log["softmax_probs_s2"].append(np.nan)  # Append NaN for s8
                    self.log["softmax_probs_s4"].append(np.nan)  # Append NaN for s8
                    self.log["second_choice_correct"].append(float(chose_correctly))
                    self.logged_choice2 = True
                    if self.log["first_choice_correct"][-1] == 0 or self.log["second_choice_correct"][-1] == 0:
                        self.log["both_choices_correct"].append(0.0)
                    else:
                        self.log["both_choices_correct"].append(1.0)
                    # if reporting:
                    #     print(f"TS:{self.num_timesteps} State:s5 Action:{self.prev_action} len1:{len(self.log['first_choice_correct'])} len2:{len(self.log['second_choice_correct'])}")

                elif snum+1 == 6:  # # Last choice made in s6 (first choice was incorrect)
                    self.log["softmax_probs_s6"].append(action_probs[0][0])  # Log softmax for s6
                    self.log["softmax_probs_s8"].append(np.nan)  # Append NaN for s5
                    self.log["softmax_probs_s2"].append(np.nan)  # Append NaN for s8
                    self.log["softmax_probs_s4"].append(np.nan)  # Append NaN for s8
                    self.log["second_choice_correct"].append(float(chose_correctly))
                    self.logged_choice2 = True
                    if self.log["first_choice_correct"][-1] == 0 or self.log["second_choice_correct"][-1] == 0:
                        self.log["both_choices_correct"].append(0.0)
                    else:
                        self.log["both_choices_correct"].append(1.0)
                    # if reporting:
                    #     print(f"TS:{self.num_timesteps} State:s6 Action:{self.prev_action} len1:{len(self.log['first_choice_correct'])} len2:{len(self.log['second_choice_correct'])}")

                elif self.prev_obs[7] == 1:  # # Last choice made in s8
                    self.log["softmax_probs_s8"].append(action_probs[0][0])  # Log softmax for s8
                    self.log["softmax_probs_s5"].append(np.nan)  # Append NaN for s5
                    self.log["softmax_probs_s6"].append(np.nan)  # Append NaN for s6
                    self.log["second_choice_correct"].append(float(chose_correctly))
                    self.logged_choice2 = True
                    if self.log["first_choice_correct"][-1] == 0 or self.log["second_choice_correct"][-1] == 0:
                        self.log["both_choices_correct"].append(0.0)
                    else:
                        self.log["both_choices_correct"].append(1.0)
                    # if reporting:
                    #     print(f"TS:{self.num_timesteps} State:s8 Action:{self.prev_action} len1:{len(self.log['first_choice_correct'])} len2:{len(self.log['second_choice_correct'])}")

        if reporting:
            print(f"TS:{self.num_timesteps} Ext:{self.extinction} State:{snum+1} Action:{self.prev_action} len1:{len(self.log['first_choice_correct'])} len2:{len(self.log['second_choice_correct'])}")

        if self.logged_choice1 and self.logged_choice2 and (not self.logged_state_values):
            # Log value estimates for all states
            self.logged_state_values = True
            for i, state_vector in enumerate(STATE_VECTORS):
                state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                with torch.no_grad():
                    value = self.model.policy.predict_values(state_tensor)  # Get critic value
                self.log["critic_values"][i].append(value.item())            

        self.prev_obs = obs  # Update the previous observation

        if self.num_timesteps == 1000000:
            self.extinction = True

        self.training_env.set_attr("prev_state", obs)

        return True  # Continue training

    def _on_training_end(self) -> None:
        # Bin results at the end of training
        self.first_choice_binned = compute_binned_averages(
            self.log["first_choice_correct"], self.bin_size
        )
        self.second_choice_binned = compute_binned_averages(
            self.log["second_choice_correct"], self.bin_size
        )
        self.both_choices_binned = compute_binned_averages(
            self.log["both_choices_correct"], self.bin_size
        )
        self.softmax_probs_s1_binned = compute_binned_averages(
            self.log["softmax_probs_s1"], self.bin_size
        )
        self.softmax_probs_s2_binned = compute_binned_averages(
            self.log["softmax_probs_s2"], self.bin_size
        )
        self.softmax_probs_s3_binned = compute_binned_averages(
            self.log["softmax_probs_s3"], self.bin_size
        )
        self.softmax_probs_s4_binned = compute_binned_averages(
            self.log["softmax_probs_s4"], self.bin_size
        )
        self.softmax_probs_s5_binned = compute_binned_averages(
            self.log["softmax_probs_s5"], self.bin_size
        )
        self.softmax_probs_s6_binned = compute_binned_averages(
            self.log["softmax_probs_s6"], self.bin_size
        )
        self.softmax_probs_s8_binned = compute_binned_averages(
            self.log["softmax_probs_s8"], self.bin_size
        )
        print(len(self.log["first_choice_correct"]))
        print(len(self.log["second_choice_correct"]))

       # Compute binned averages for critic values
        self.critic_values_binned = {}
        for state_index, values in self.log["critic_values"].items():
            print(f'{state_index} {len(values)}')
            self.critic_values_binned[state_index] = compute_binned_averages(values, self.bin_size)
            #print(f'{state_index} {len(self.critic_values_binned[state_index])}')


def compute_binned_averages(values, bin_size):
    num_bins = len(values) // bin_size
    return [
        np.nanmean(values[i * bin_size : (i + 1) * bin_size]) * 100  # Use np.nanmean
        for i in range(num_bins)
    ]

class CustomTaskEnv(gym.Env):
    def __init__(self, version=1):
        super(CustomTaskEnv, self).__init__()
        
        # Environment configuration
        self.callback = None
        self.version = version
        self.state_space = 32  # s1 - s32
        self.action_space = gym.spaces.Discrete(4)  # L, R, F, B
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_space,), dtype=np.float32)
        
        self.state = None
        self.prev_state = None
        self.extinction = False
        self.done = False
        self.transitions = None
        self.valvec = None

    def set_callback(self, callback):
        """Assign the callback reference."""
        self.callback = callback

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new trial."""
        self.state = np.zeros(self.state_space, dtype=np.float32)
        if self.version == 1:
            self.start_state = np.random.choice([0, 2])  # Randomly choose between s1 and s2
            if self.start_state==0:
                self.transitions = TRANSITIONS_S1
            else:
                self.transitions = TRANSITIONS_S3

        else:
            self.start_state = np.random.choice([0, 3])  # Randomly choose between s1 and s2
        self.state[self.start_state] = 1
        self.done = False
        return self.state, {}

    def step(self, action):
        """Take an action and observe the outcome."""
        reward = 0
        truncated = False  # No truncation logic for this environment

        # Access the property from the callback
        extinction = self.callback.extinction

        if self.done:
            raise ValueError("Episode has already ended. Call reset() to start a new episode.")
        
        first_nonzero_index = next((index for index, value in enumerate(self.state) if value != 0), None)

        self.prev_state = self.state

        print(f'{snum} {first_nonzero_index} {action} {1+self.transitions[first_nonzero_index][action]}')
        print(self.valvec)
    
        self.state = np.array(STATE_VECTORS[self.transitions[first_nonzero_index][action]], dtype=np.float32)

        snum = next((index for index, value in enumerate(self.state) if value != 0), None) 

        if not extinction:  # self.prev_state is set in _on_step
            if self.prev_state[25] == 1 and snum in (24,25):  # earned a reward in s8
                self.done = True
                reward = 1       
                print("reward")
        else:
            if self.prev_state[29] == 1 and snum in (28,29):  # earned a reward in s8
                self.done = True
                reward = 1       
                print("ext reward")

        if first_nonzero_index in (24, 25, 26, 27, 28, 29, 30, 31) and snum in (24, 25, 26, 27, 28, 29, 30, 31):  # ended trial in a corner   
            self.done = True

        return self.state, reward, self.done, truncated, {}

# Number of training runs
N = 8
bin_size = 32
version_flag = 1  # Set to 1 or 2 depending on the desired task version

# Placeholder for binned results across all runs
all_first_choice_binned = []
all_second_choice_binned = []
all_both_choices_binned = []

# Run the training process N times
for i in range(N):
    print(f"Training run {i + 1} / {N}...")

    # Initialize environment
    env = CustomTaskEnv(version=version_flag)

    policy_kwargs = dict(
        net_arch=[8],  # Define your network architecture
    )

    # Initialize PPO with the default policy and custom architecture
    callback = ChoiceLoggingCallback(env, bin_size=32)
    env.set_callback(callback)

    check_env(env)  # Verify the custom environment

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=16,
        batch_size=4,
        policy_kwargs=policy_kwargs,
        verbose=0,
        gamma=0.3,
        learning_rate=0.0005,
    )

    # Train PPO model
    model.learn(total_timesteps=960*4,callback=callback)
    
    # Store binned results for post-processing
    all_first_choice_binned.append(callback.first_choice_binned)
    all_second_choice_binned.append(callback.second_choice_binned)
    all_both_choices_binned.append(callback.both_choices_binned)

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))  # 3 rows, 1 column of subplots

    # Plot choice probabilities in the first subplot
    axes[0].plot(
        range(1, len(callback.first_choice_binned) + 1),
        callback.first_choice_binned,
        label="First Choice Correct (s1, s2)",
        marker="o",
    )
    axes[0].plot(
        range(1, len(callback.second_choice_binned) + 1),
        callback.second_choice_binned,
        label="Second Choice Correct (s3)",
        marker="o",
    )
    axes[0].plot(
        range(1, len(callback.both_choices_binned) + 1),
        callback.both_choices_binned,
        label="Correct Trial",
        marker="o",
    )
    axes[0].set_title("Correct Choices")
    axes[0].set_xlabel("Session Number")
    axes[0].set_ylabel("% Correct")
    axes[0].legend()
    axes[0].grid()

    # Plot softmax probabilities in the second subplot
    axes[1].plot(
        range(1, len(callback.softmax_probs_s1_binned) + 1),
        callback.softmax_probs_s1_binned,
        label="P(a1) in s1",
        marker="x",
    )
    axes[1].plot(
        range(1, len(callback.softmax_probs_s3_binned) + 1),
        callback.softmax_probs_s3_binned,
        label="P(a1) in s3",
        marker="x",
    )
    axes[1].plot(
        range(1, len(callback.softmax_probs_s8_binned) + 1),
        callback.softmax_probs_s8_binned,
        label="P(a1) in s8",
        marker="x",
    )
    axes[1].plot(
        range(1, len(callback.softmax_probs_s6_binned) + 1),
        callback.softmax_probs_s6_binned,
        label="P(a1) in s6",
        marker="x",
    )
    axes[1].set_title("Binned Softmax Probabilities")
    axes[1].set_xlabel("Bin (10 Trials per Bin)")
    axes[1].set_ylabel("Softmax Probability (%)")
    axes[1].legend()
    axes[1].grid()

    desired_state_indices = {0, 3, 7, 25, 27, 29, 31}

    # Plot critic value binned averages in the third subplot
    for state_index, binned_values in callback.critic_values_binned.items():
        if state_index in desired_state_indices:
            axes[2].plot(
                range(1, len(binned_values) + 1),
                binned_values,
                label=f"Critic Value for State {state_index}",
                marker="*",
            )
    axes[2].set_title("Binned Critic Values for All States")
    axes[2].set_xlabel("Bin (10 Trials per Bin)")
    axes[2].set_ylabel("Critic Value")
    axes[2].legend()
    axes[2].grid()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def find_end_bins(all_binned_data):
    # Process and realign each binned vector
    end_bins = []

    for binned_data in all_binned_data:
        # Identify the bin where both_choices_binned > 0.7 for the second time in a row
        for i in range(1, len(binned_data)):
            if binned_data[i - 1] >= 70 and binned_data[i] >= 70:
                end_bin = i
                break
        else:
            end_bin = len(binned_data) - 1  # If no such bin is found, include all bins
        end_bins.append(end_bin)

    return end_bins

def process_and_pad_binned_data(all_binned_data, end_bins):
    # Process and realign each binned vector
    processed_binned_data = []
    max_length = 0  # Track the global maximum length for alignment

    for binned_data, end_bin in zip(all_binned_data, end_bins):

        # Replace bins after `end_bin` with NaNs
        processed_data = copy.deepcopy(binned_data)
        processed_data[end_bin + 1:] = [np.nan] * (len(binned_data) - end_bin - 1)

        # Append the processed data
        processed_binned_data.append(processed_data)

        # Update the global maximum length
        max_length = max(max_length, len(processed_data))

    # Align and pad all rows to the maximum length
    aligned_data = []
    for data in processed_binned_data:
        # Ensure `data` is a list or array
        data = list(data)
        
        # Calculate the necessary padding at the start
        padding = [np.nan] * (max_length - len(data))
        
        # Append the padding at the start and valid data at the end
        padded_data = padding + data  # Prepend padding
        aligned_data.append(padded_data)

    return np.array(aligned_data)

# Process (align) all binned data
end_bins = find_end_bins( all_both_choices_binned)
processed_first_choice_binned = process_and_pad_binned_data(all_first_choice_binned, end_bins)
processed_second_choice_binned = process_and_pad_binned_data(all_second_choice_binned, end_bins)
processed_both_choices_binned = process_and_pad_binned_data(all_both_choices_binned, end_bins)

def shift_nans_to_left(data):
    shifted_data = []
    for row in data:
        row = np.array(row)  # Ensure it's a NumPy array
        while np.isnan(row[-1]):  # Keep shifting until the last element is not NaN
            row = np.roll(row, 1)
        shifted_data.append(row)
    return np.array(shifted_data)


# Shift NaNs to the left
processed_first_choice_binned = shift_nans_to_left(processed_first_choice_binned)
processed_second_choice_binned = shift_nans_to_left(processed_second_choice_binned)
processed_both_choices_binned = shift_nans_to_left(processed_both_choices_binned)

def align_and_pad_all_binned_data(*all_binned_data):
    # Find the global maximum length across all vectors
    max_length = max(len(data) for binned_data in all_binned_data for data in binned_data)
    
    # Align and pad each vector within each group of binned data
    aligned_data = []
    for binned_data in all_binned_data:
        aligned_group = []
        for data in binned_data:
            # Ensure `data` is a NumPy array
            data = np.array(data, dtype=np.float32)
            
            # Calculate padding needed at the start
            padding = np.full(max_length - len(data), np.nan, dtype=np.float32)
            
            # Prepend padding to align the vector to the end
            padded_data = np.concatenate((padding, data))
            aligned_group.append(padded_data)
        
        aligned_data.append(np.array(aligned_group))
    
    return aligned_data


# Align and pad all processed binned data
aligned_first_choice_binned, aligned_second_choice_binned, aligned_both_choices_binned = align_and_pad_all_binned_data(
    processed_first_choice_binned,
    processed_second_choice_binned,
    processed_both_choices_binned
)

# Compute averages across all runs
average_first_choice_binned = np.nanmean(aligned_first_choice_binned, axis=0)
average_second_choice_binned = np.nanmean(aligned_second_choice_binned, axis=0)
average_both_choices_binned = np.nanmean(aligned_both_choices_binned, axis=0)

## Test the trained model
# def test_agent(env, model, episodes=10):
#     for episode in range(episodes):
#         state, _ = env.reset()
#         done = False
#         total_reward = 0

#         while not done:
#             action, _states = model.predict(state)
#             state, reward, done, truncated, _ = env.step(action)
#             total_reward += reward

#         print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# test_agent(env, model)

# print(len(callback.first_choice_binned))
# print(len(callback.second_choice_binned))
# print(len(callback.log['first_choice_correct']))
# print(len(callback.log['second_choice_correct']))

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(
    range(1, len(average_first_choice_binned) + 1),
    average_first_choice_binned,
    label="First Choice Correct",
    marker="o",
)
ax.plot(
    range(1, len(average_second_choice_binned) + 1),
    average_second_choice_binned,
    label="Second Choice Correct",
    marker="o",
)
ax.plot(
    range(1, len(average_both_choices_binned) + 1),
    average_both_choices_binned,
    label="Correct Trial",
    marker="o",
)

ax.set_title("Average Correct Choices Across Training Runs")
ax.set_xlabel("Bin (Aligned to End at Threshold)")
ax.set_ylabel("Percentage Correct (%)")
ax.legend()
ax.grid()

plt.tight_layout()
plt.show()