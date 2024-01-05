import os, sys, time, subprocess  
import numpy as np 
from MonkeyMazeEnv import MonkeyMazeEnv 
from exploration_reward_env_train.MonkeyMazeEnvExplore import MonkeyMazeEnvExplore 
from agent_package.valina_dqn_agent import Vanila_Agent
from path_analytic_tool.path_similarity import path_similarity
from path_analytic_tool.choice_sim_percentage import choice_sim_percent
from path_analytic_tool.shannon_info import shannon_info
from path_analytic_tool.DataSaver import DataSaver 
from path_analytic_tool.exploration_entropy import exploration_entropy 
from plotting_functions.single_q_arrow_show import single_q_arrow_show
from path_analytic_tool.monkey_max_choice_agent_compare import monkey_max_choice_agent_compare
from MonkeyPath import MonkeyPath 
from plotting_functions.q_values_map import q_values_map 
from torch.utils.tensorboard import SummaryWriter



def pre_train(agent=Vanila_Agent, episode_num = 10):
    max_episode_num = 100 
    env = MonkeyMazeEnv(no_reward=True)
    env.max_episode_step = max_episode_num 
  

    for i_episode in range(-1, -episode_num-1, -1):
        state, info = env.reset() 
        done = False 
        truncated = False 
        total_length = 1 
        total_reward = 0 
    
        while not(done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward 
            total_length += 1



            agent.remember(state, action, reward, next_state, done)

            
            state = next_state 
    
            agent.replay() 

        agent.decay_epsilon() 
        print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(i_episode, total_reward, agent.epsilon, total_length))
    
    agent.epsilon = 0.8
      

explore_reward = False 
    
monkey_name = "p" 
bool_pre_train = True 
bool_PER = False 
env = MonkeyMazeEnv() 
monkey_path = MonkeyPath(monkey_name=monkey_name)
trial_length = monkey_path.trial_num
game_name = "Valina_dqn" if not(explore_reward) else "Valina_dqn_explore"
run_time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
log_dir = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/tensorboard_Data/')
data_dir = os.path.join(os.path.join(os.path.expanduser('~'), f'Desktop/model_Data/{game_name}/{monkey_name}/{str(run_time)}/'))
if not(os.path.exists(data_dir)):
    os.makedirs(data_dir)

## tensorboard setting 
port = 6145
subprocess.Popen(f"tensorboard --logdir={log_dir} --port={port} --reload_multifile=true", shell=True)
log_dir = log_dir + f"{game_name}/{monkey_name}/{str(run_time)}"
log_text_dir = log_dir + "log.txt"
writer = SummaryWriter(log_dir = log_dir)

if not(os.path.exists(log_text_dir)):
    with open(log_text_dir, 'w') as log_file:
        log_file.write("Training log of monkey maze env \n")


task_info = f"Task at time: {str(run_time)} is vanilla dqn training with pre_training = {str(bool_pre_train)} and per {str(bool_PER)}\n"

with open(log_text_dir, 'a') as log_file:
    log_file.write(task_info + '\n') 

data_saver = DataSaver(data_dir = data_dir, model_name = game_name) 

## agent_setting 
state_size = env.state_n
action_size = env.action_n
hidden_size = 512
learning_rate = 0.001
memory_size = 10000
batch_size = 128
gamma = 0.99

agent = Vanila_Agent(state_size = state_size, 
                     action_size = action_size,
                     hidden_size = hidden_size,
                     learning_rate = learning_rate,
                     memory_size = memory_size,
                     batch_size = batch_size, 
                     gamma = gamma,)


if bool_pre_train:
    pre_train(agent = agent, episode_num = 10)

for i in range(100):
    trial_num = 190 
    trial_start = monkey_path.get_start_position(trial_num) 
    trial_goal = monkey_path.get_goal_position(trial_num)
    trial_monkey_path = monkey_path.get_trial(trial_num = trial_num)
    monkey_agent_compare_gif_dir = data_dir + f"monkey_compare_agent/"
    if not(os.path.exists(monkey_agent_compare_gif_dir)):
        os.makedirs(monkey_agent_compare_gif_dir)

    if explore_reward: 
        env = MonkeyMazeEnvExplore(render_mode = "human", file_name = monkey_agent_compare_gif_dir + f"trial_{str(trial_num)}.gif") 
    else:
        env = MonkeyMazeEnv(render_mode = "human", file_name = monkey_agent_compare_gif_dir + f"trial_{str(trial_num)}.gif") 

    trial_path_max = len(trial_monkey_path)
    trial_t = 0
    env._monkey_location = trial_start
    state, info = env.reset(reward = trial_goal, start = trial_start)

    reward = 0
    done = False 
    truncated = False 
    total_length = 1
    total_reward = 0 

    state_trajectories = []
    action_trajectories = [] 

    while not(done or truncated):
        trial_t += 1 

        env._monkey_location = trial_monkey_path[min(trial_t, trial_path_max - 1)]

        action = agent.act(state)

        next_state, reward, done, truncated, info = env.step(action) 

        total_reward += reward 
        total_length += 1 

        agent.remember(state, action, reward, next_state, done)

        

        agent.replay(TD_sample=bool_PER)

        state_trajectories.append(state)
        action_trajectories.append(action)
        state = next_state

        single_q_arrow_show(agent, trial_num, trial_t, state, trial_goal, data_dir)
        q_values_map(agent, trial_num, trial_t, state, trial_goal, data_dir)

    
    if done:
        agent.decay_epsilon()
    
    while trial_t < trial_path_max - 1:
        trial_t += 1 
        env._monkey_location = trial_monkey_path[trial_t]
        env._render_frame() 

    shannon_value = shannon_info(state_trajectories, action_trajectories, env.action_n)
    path_sim = path_similarity(trial_monkey_path, state_trajectories)
    choice_sim_value = choice_sim_percent(agent, trial_monkey_path)
    monkey_max_choice_compare_percent = monkey_max_choice_agent_compare(trial_monkey_path, agent)
    exploration_entropy_value, exploration_percentage = exploration_entropy(state_trajectories) 


    writer.add_scalar("reward", total_reward, trial_num) 
    writer.add_scalar("length", total_length, trial_num)
    writer.add_scalar("reward_rate", total_reward/total_length, trial_num)
    writer.add_scalar("epsilion", agent.epsilon, trial_num)
    writer.add_scalar("shannon_value:", shannon_value, trial_num)
    writer.add_scalar("path_sim", path_sim, trial_num) 
    writer.add_scalar("choice_sim_percentage" , choice_sim_value, trial_num) 
    writer.add_scalar("monkey_max_choice_agent_compare", monkey_max_choice_compare_percent, trial_num)
    writer.add_scalar("exploration_entropy", exploration_entropy_value, trial_num)
    writer.add_scalar("exploration_percentage", exploration_percentage, trial_num)


    data_saver.record_data(length = total_length,
                            reward = total_reward,
                            reward_rate = total_reward/total_length,
                            shannon_value = shannon_value,
                            path_sim = path_sim,
                            choice_sim_percent = choice_sim_value,
                            monkey_max_choice_compare_percent = monkey_max_choice_compare_percent,
                            exploration_entropy = exploration_entropy_value,
                            explore_percentage = exploration_percentage, 
                            trajectory = state_trajectories)

    print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(trial_num, total_reward, agent.epsilon, total_length))
    env.recrdr.save()
    env.close()

writer.close() 
data_saver.save_data() 



    
