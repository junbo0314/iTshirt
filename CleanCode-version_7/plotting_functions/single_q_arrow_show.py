import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors 
import matplotlib
import matplotlib.cm as cm 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from agent_package.ModelBasedAgent import ModelBasedAgent as Agent  
import pickle


def single_q_arrow_show(agent = Agent(1,1,1,1,1,1,1), i_episode = 0, counter = 0, state=np.zeros([0,0]), reward_location = np.zeros([0,0]), data_dir = "network_img/"):
 
    plt.switch_backend('agg')
    data_dir = data_dir + f"network_image/{str(i_episode)}/"

    if not(os.path.exists(data_dir)):
        os.makedirs(data_dir)

    file_name = data_dir + f"{str(counter)}.jpg"

    with open("env_map", 'rb') as fp:
      map = pickle.load(fp)

    map[reward_location[0]][reward_location[1]] = 0.5 

    


    

    arrow_map = np.ones([11, 11])*10


    arrows = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
        }

    map_positions = np.transpose(np.where(map > 0))

    for node in map_positions:
        agent_choice = agent.act(node, max_option=True)
        arrow_map[node[0]][node[1]] = agent_choice
        

    fig, ax = plt.subplots(figsize=(11, 11))
    scale = 0.25 

    for r, row in enumerate(arrow_map):
        for c, cell in enumerate(row):
            if cell == 10:
                continue
            else:
                agent_arrows = plt.arrow(c, r, scale*arrows[cell][0], scale*arrows[cell][1], head_width = 0.15, color = 'r', label='agent')
    

    agent_state = plt.Circle((state[1], state[0]), 0.3, color = 'dodgerblue')
    ax.add_patch(agent_state)
    plt.legend(handles = [agent_arrows]) 
    plt.imshow(map, cmap=cm.Greys)
    plt.savefig(file_name)
    plt.cla() 
    plt.close(fig=fig)

    with open(data_dir + f"arrow_matrix_{str(counter)}", 'wb') as fp:
                        pickle.dump(arrow_map, fp)





    
     