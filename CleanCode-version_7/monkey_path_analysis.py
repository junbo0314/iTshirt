import numpy as np 
from MonkeyMazeEnv import MonkeyMazeEnv 
from MonkeyPath import MonkeyPath 
from path_analytic_tool.exploration_entropy import exploration_entropy
from path_analytic_tool.shannon_info import shannon_info 
from path_analytic_tool.DataSaver import DataSaver 


monkey_name = "p" 
monkey_path = MonkeyPath(monkey_name=monkey_name) 
trial_num = monkey_path.trial_num 

data_dir = f"monkey_path_data/" 

data_saver = DataSaver(data_dir=data_dir, model_name=f"{monkey_name}_path_data") 


action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

direction_to_action = {
            (1, 0): 0,
            (0, 1): 1,
            (-1, 0): 2, 
            (0, -1): 3,
        }   


env = MonkeyMazeEnv()

for i in range(trial_num): 
    start = monkey_path.get_start_position(i)
    goal = monkey_path.get_goal_position(i)
    state, info = env.reset(reward=goal, start=start)
    env._monkey_location = start
    done = False 
    truncated = False 
    total_length = 1 
    total_reward = 0 
    
    trial_monkey_path = monkey_path.get_trial(i) 
    trial_path_max = len(trial_monkey_path) 
    t = 0 

    state_trajectories = [] 
    action_trajectories = [] 



    while not(done or truncated):
        if not(t >= trial_path_max ): 
            state_trajectories.append(env._monkey_location) 
        

        t += 1
         
        action_vec = np.float64(trial_monkey_path[min(t, trial_path_max-1)]) - np.float64(env._monkey_location)
        env._monkey_location = trial_monkey_path[min(t, trial_path_max-1)]

        if not(t >= trial_path_max ):
            action = direction_to_action[tuple(action_vec)] 
        action_trajectories.append(action) 
        env._agent_location = env._monkey_location 
        small_rewarded = env.check_sub_reward()
        if small_rewarded:
            total_reward += 1 
    
        total_length += 1
        if t >= trial_path_max:
            done = True

    explore_entropy, explore_percentage = exploration_entropy(trial_monkey_path)

    total_reward += 8
    shannon_value = shannon_info(state_trajectories, action_trajectories, env.action_n)
    data_saver.record_data(total_length, total_reward, total_reward/total_length, 
                           shannon_value, 1, 1, 1, explore_entropy, explore_percentage)
    

data_saver.save_data() 