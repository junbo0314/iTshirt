
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors 
import matplotlib
import matplotlib.cm as cm 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from agent_package.ModelBasedAgent import ModelBasedAgent as Agent 
import pickle
import torch 


def q_values_map(agent = Agent(1,1,1,1,1,1,1), i_episode = 0, counter = 0, state=np.zeros([0,0]), reward_location = np.zeros([0,0]), data_dir = "network_img/"):
 
    plt.switch_backend('agg')
    data_dir = data_dir + f"Q_value_map/{str(i_episode)}/"

    if not(os.path.exists(data_dir)):
        os.makedirs(data_dir)

    file_name = data_dir + f"{str(counter)}.jpg"

    with open("env_map", 'rb') as fp:
      map = pickle.load(fp)

    map[reward_location[1]][reward_location[0]] = 0.5
    map_positions = np.transpose(np.where(map > 0))
    Q_Values = np.zeros([11, 11])
    
    vmin = 1
    vmax = 10 

    for state_place in map_positions:
        Q_Values[state_place[0]][state_place[1]] = np.max(agent.q_network(torch.Tensor(state).to(agent.device)).cpu().detach().numpy()) 


    fig, ax = plt.subplots(figsize=(11, 11))
    scale = 0.25 

   

    agent_state = plt.Circle((state[1], state[0]), 0.3, color = 'dodgerblue')
    plt.scatter(reward_location[1], reward_location[0], marker='*', s=100, color='blue')

    ax.add_patch(agent_state)
    heatmap = plt.imshow(Q_Values, cmap=cm.Reds, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Values')
    plt.savefig(file_name)
    plt.cla() 
    plt.close(fig=fig)



if __name__ == "__main__":
    agent = Agent(2,4,256,0.001,1000, 64, 0.55, explore_agent=True)
    q_values_map(agent=agent, i_episode=0, counter=0, state=np.array([0,0]), reward_location=np.array([10,10]))

    
     

