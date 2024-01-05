import numpy as np
import sys, os 
import pickle
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from MonkeyMazeEnv import MonkeyMazeEnv
from MonkeyPath import MonkeyPath


def plot_single_monkey_trial_to_gif(trial_num = 190):
    if trial_num is None:
        trial_num = 1 

    env = MonkeyMazeEnv(render_mode="human", file_name="example_monkey.gif") 
    


    
    env.monkey_only = True 
    monkey_path = MonkeyPath() 
    trial_seq = monkey_path.get_trial(trial_num)
    start_pos = monkey_path.get_start_position(trial_num) 
    goal_pos = monkey_path.get_goal_position(trial_num)


    env.reset(reward = goal_pos, start = start_pos)

    for state in trial_seq:
        env._agent_location = np.array(state)
        env._monkey_location = np.array(state)
        env.check_sub_reward() 
        env._render_frame() 
    env.recrdr.save() 


if __name__ == "__main__":
    plot_single_monkey_trial_to_gif() 
    
