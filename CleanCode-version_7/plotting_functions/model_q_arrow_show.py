import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from agent_package.ModelBasedAgent import ModelBasedAgent as Agent
from agent_package.CuriosityCounterModelTable import CuriosityCounterModelTable
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors 
import matplotlib
import matplotlib.cm as cm 
import os 


def model_q_arrow_show(agent = Agent(1,1,1,1,1,1,1), model = CuriosityCounterModelTable(), i_episode = 0, counter = 0, state=np.zeros([0,0]), data_dir="img/"):

    aligned_counter = 0 
    plt.switch_backend('agg')
    data_dir = data_dir + f"network_image/{str(i_episode)}/"

    if not(os.path.exists(data_dir)):
        os.makedirs(data_dir)

    file_name = data_dir + f"{str(counter)}.jpg"
    cognitve_map = np.copy(model.model_map)
    node_num = np.sum(cognitve_map)


    if model.known_reward() and not(model.reward_location is None):
        reward_location = model.reward_location
        cognitve_map[reward_location[0]][reward_location[1]] = 0.7

    exploit_map = np.ones([model.env_size, model.env_size])*10
    explore_map = np.ones([model.env_size, model.env_size])*10 


    arrows = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
        }

    map_positions = np.transpose(np.where(cognitve_map > 0))

    for node in map_positions:
        exploit_choice = agent.act(node, max_option=True)
        explore_choice = agent.explore_act(node, max_option=True) 

        exploit_map[node[0]][node[1]] = exploit_choice
        explore_map[node[0]][node[1]] = explore_choice

    fig, ax = plt.subplots(figsize=(model.env_size, model.env_size))
    scale = 0.25 

    for r, row in enumerate(exploit_map):
        for c, cell in enumerate(row):
            if cell == 10:
                continue
            else:
                exploit_arrows = plt.arrow(c, r, scale*arrows[cell][0], scale*arrows[cell][1], head_width = 0.15, color = 'r', label='exploit')
    
    for r, row in enumerate(explore_map):
        for c, cell in enumerate(row):
            if cell == 10:
                continue
            elif exploit_map[r][c] == cell:
                aligned_counter += 1 
                same_arrows = plt.arrow(c, r, scale*arrows[cell][0], scale*arrows[cell][1], head_width = 0.15, color = 'g', label='aligned')

 
            else:
                explore_arrows = plt.arrow(c, r, scale*arrows[cell][0], scale*arrows[cell][1], head_width = 0.15, color = 'b', label='explore')

    agent_state = plt.Circle((state[1], state[0]), 0.3, color = 'dodgerblue')
    ax.add_patch(agent_state)
    plt.imshow(cognitve_map, cmap=cm.Greys)

    
    if aligned_counter > 0:
        plt.legend(handles = [exploit_arrows, same_arrows, explore_arrows])
    else:
        plt.legend(handles = [exploit_arrows, explore_arrows])

    plt.savefig(file_name)
    plt.cla() 
    plt.close(fig=fig)

    return (aligned_counter/node_num)*100 




    
     