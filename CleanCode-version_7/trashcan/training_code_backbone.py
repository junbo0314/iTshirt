import numpy as np
import os, time, subprocess, sys
from torch.utils.tensorboard import SummaryWriter
from path_analytic_tool.choice_sim_percentage import *
from path_analytic_tool.path_similarity import *
from path_analytic_tool.shannon_info import *
from path_analytic_tool.monkey_max_choice_agent_compare import monkey_max_choice_agent_compare
from plotting_functions.single_q_arrow_show import *
from MonkeyMazeEnv import MonkeyMazeEnv
from torch.utils.tensorboard import SummaryWriter 

## implement model if needed 
def pre_train(agent = Agent(1,1,1,1,1,1,1), model = None, episode_num = 10): 
    max_episode = 100 
    env = MonkeyMazeEnv(no_reward = True)
    env.max_episode_step = max_episode 

    for i_episode in range(-1, -episode_num-1, -1):
        state, info = env.reset() 

        """ if model is implemented, reset model here
        model.reset() 
        model.t = 0
        model.draw_map(i_episode)
        """
        
        done = False 
        truncated = False 
        total_length = 1
        total_reward = 0 
        
        # if model is implemented, record map here
        # model.record_map(state = state, reward = 0, done = done, i_episode = i_episode)

        while not(done or truncated): 
            total_length += 1 

            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)

            if reward >= 0:
                reward = 0

            total_reward += reward 

            agent.remember(state, action, reward, next_state, done) 

            # if model is implemented, record map here
            # model.record_map(state = next_state, reward = reward, done = False, i_episode = i_episode)


            state = next_state
            # if model is implemented, model simulate here (offline learning using model simulation)
            # model.model_simulate(agent=agent, state = state)

            agent.replay() 
        
        agent.decay_epsilon() 
        print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(i_episode, total_reward, agent.epsilon, total_length))
    
    agent.epsilon = 0.8 


monkey_name = "p" 
bool_pre_train = True 
bool_PER = False 
env = MonkeyMazeEnv() 
monkey_path = MonkeyPath(monkey_name = monkey_name)
trial_length = monkey_path.trial_num
## change the game name for data saving directory 
game_name = "testy_test_test"
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
        log_file.write("Training log of monkey maze env\n")

## log the task info 
task_info = f"Task at time: {str(run_time)} is {game_name} training with pre_training = {str(bool_pre_train)} and per {str(bool_PER)}\n"

with open(log_text_dir, 'a') as log_file:
    log_file.write(task_info + '\n') 

## agent_setting 
state_size = env.state_n
action_size = env.action_n
hidden_size = 512
learning_rate = 0.001
memory_size = 10000
batch_size = 128
gamma = 0.99

agent = Agent(state_size = state_size,
              action_size = action_size, 
              hidden_size = hidden_size,
              learning_rate = learning_rate,
              memory_size = memory_size,
              batch_size = batch_size,
              gamma = gamma)

# if model is implemented, initialize model here
model = None 

if bool_pre_train:
    pre_train(agent = agent, model = model)


for trial_num in range(trial_length):
    trial_start = monkey_path.get_start_position(trial_num) 
    trial_goal = monkey_path.get_goal_position(trial_num)
    trial_monkey_path = monkey_path.get_trial(trial_num = trial_num)
    monkey_agent_compare_gif_dir = data_dir + f"monkey_compare_agent/"
    if not(os.path.exists(monkey_agent_compare_gif_dir)):
        os.makedirs(monkey_agent_compare_gif_dir)

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

    """ if model is implemented, reset model here
    model.reset(reset_short_term = True)
    model.t = 0 
    model.draw_map(episode_num = trial_num)
    model.record_map(state = state, reward = reward, done = False, i_episode = trial_num)
    """
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
        model.record_map(state = next_state , reward = reward, done = done, i_episode = trial_num)

        

        agent.replay(TD_sample=bool_PER)
        

        state_trajectories.append(state)
        action_trajectories.append(action)
        state = next_state
        # if model is implemented, model simulate here (offline learning using model simulation)
        # model.model_simulate(agent = agent, state = state, reset = True)

        single_q_arrow_show(agent, trial_num, trial_t, state, trial_goal, data_dir)

    if done:
        agent.decay_epsilon() 

    while trial_t < trial_path_max-1:
        trial_t += 1 
        env._monkey_location = trial_monkey_path[trial_t]
        env._render_frame() 

    
    shannon_value = shannon_info(state_trajectories, action_trajectories, env.action_n)
    path_sim = path_similarity(trial_monkey_path, state_trajectories)
    choice_sim_value = choice_sim_percent(agent, trial_monkey_path)
    monkey_max_choice_compare_percent = monkey_max_choice_agent_compare(trial_monkey_path, agent)


    writer.add_scalar("reward", total_reward, trial_num) 
    writer.add_scalar("length", total_length, trial_num)
    writer.add_scalar("reward_rate", total_reward/total_length, trial_num)
    writer.add_scalar("epsilion", agent.epsilon, trial_num)
    writer.add_scalar("shannon_value:", shannon_value, trial_num)
    writer.add_scalar("path_sim", path_sim, trial_num) 
    writer.add_scalar("choice_sim_percentage" , choice_sim_value, trial_num) 
    writer.add_scalar("monkey_max_choice_agent_compare", monkey_max_choice_compare_percent, trial_num)

    print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(trial_num, total_reward, agent.epsilon, total_length))
    env.recrdr.save()
    env.close()

writer.close() 
    

        



            

        

