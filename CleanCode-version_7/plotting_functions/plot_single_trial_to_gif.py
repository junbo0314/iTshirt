import numpy as np
import sys, os 
import pickle
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from MonkeyMazeEnv import MonkeyMazeEnv

def plot_single_trial_to_gif(trial_seq = []):
    env = MonkeyMazeEnv(render_mode="human", file_name="example.gif")
    if len(trial_seq) == 0: 
        with open("/Users/swankim/Desktop/Excool_Assignment/CleanCode-version_2/plotting_functions/example_trajectories", "rb") as file:
            trial_seq = pickle.load(file)
    
    trial_seq = np.array(trial_seq)


    env.reset(reward = trial_seq[-1,:], start = trial_seq[0,:])
    trial_len = np.size(trial_seq, 0)
    
    for state in trial_seq:
        env._agent_location = state
        env.check_sub_reward()
        env._render_frame() 

    env.recrdr.save()

    env.close() 


if __name__ == "__main__":
    plot_single_trial_to_gif() 
    

