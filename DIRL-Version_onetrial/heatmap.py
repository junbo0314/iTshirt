import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from MonkeyPath import *


def make_value_function_heatmap() :
    for i in range(50,1550,50) :
        value_function_file = pd.read_csv(address+"/value_function_"+str(i)+".csv")
        value_map = value_function_file.to_numpy()
        maxv = np.max(value_map)
        minv = 0
        heatmap = sns.heatmap(value_function_file, vmin = minv, vmax = maxv, cmap='YlGnBu')
        plt.title("value_function_"+str(i))
        plt.plot()
        plt.savefig(address+'/value_function_'+str(i)+'.png')
        plt.close()

def make_reward_heatmap() :
    reward_file = pd.read_csv(address+"/20ep/day2/reward_400_21"+".csv")
    reward_map = reward_file.to_numpy()
    maxv = 2.7
    minv = 0
    heatmap = sns.heatmap(reward_file, vmin = minv, vmax = maxv, cmap='YlGnBu')
    plt.title("/20ep/day2/reward_400_21")
    plt.plot()
    plt.savefig(address+"/20ep/day2/reward_400_21"+'.png')
    plt.close()

def make_dirl_heatmap(i,address) :
    reward_file = pd.read_csv(address+"/time"+str(i)+".csv")
    reward_map = reward_file.to_numpy()
    maxv = 2.7
    minv = 0
    heatmap = sns.heatmap(reward_file, vmin = minv, vmax = maxv, cmap='YlGnBu',
                          xticklabels=False, yticklabels=False,
                          cbar_kws={'label': 'Reward'})
    if i == 0 :
        plt.title("When the monkey respawns")
    plt.plot()
    plt.savefig(address+'/time'+str(i)+'.png')
  
    plt.close()
def make_dirl_goal_map(i,address) :
    goal_file = pd.read_csv(address+"/goal"+str(i)+".csv")
    reward_map = goal_file.to_numpy()
    maxv = 2.7
    minv = 0
    heatmap = sns.heatmap(goal_file, vmin = minv, vmax = maxv, cmap='YlGnBu',
                          xticklabels=False, yticklabels=False,
                          cbar_kws={'label': 'Reward'})
    plt.title("goal"+str(i))
    plt.plot()
    plt.savefig(address+'/goal'+str(i)+'.png')
    plt.close()

monkey = 'p'

n_maps = 3
bias_lam=[25,25,25]
lr_maps=0.001
monkey_path=MonkeyPath()
day=2




if __name__ == "__main__" :
    trials=monkey_path.new_pos

    for i in range(len(trials)-1):
        for j in range(len(trials[i+1])):
            address = '/Users/simjunbo/Desktop/CleanCode-version_2/irl/'+monkey+'/day'+str(i+2)+'/'+str(j)+\
        '_map'+str(n_maps)+'_bias'+str(bias_lam)
            step=trials[i+1][j][0]

            for l in range(len(step)-1):
                make_dirl_heatmap(l,address)

            for k in range(n_maps):
                make_dirl_goal_map(k,address)
        
#알아볼   것:monkey_path.new_pos,len(new pos), new pos[0]