import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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


def make_dirl_heatmap(i, maxv) :
    reward_file = pd.read_csv(address+"/time"+str(i)+".csv")
    reward_map = reward_file.to_numpy()
    maxv = 3.3
    minv = 0
    heatmap = sns.heatmap(reward_file, vmin = minv, vmax = maxv, cmap='YlGnBu',
                          xticklabels=False, yticklabels=False,
                          cbar_kws={'label': 'Reward'})
    plt.title("time"+str(i+1))
    plt.plot()
    plt.savefig(address+"/time"+str(i)+'.png')
    plt.close()

def make_dirl_goal_map(i, maxv) :
    goal_file = pd.read_csv(address+"/goal"+str(i)+".csv")
    reward_map = goal_file.to_numpy()
    maxv = 2.6
    minv = 0
    heatmap = sns.heatmap(goal_file, vmin = minv, vmax = maxv, cmap='YlGnBu',
                          xticklabels=False, yticklabels=False,
                          cbar_kws={'label': 'Reward'})
    if i==0 :
        title = "Current Goal"
    elif i==1 :
        title = "Old Goal"
    else :
        title = "Subgoal"
    plt.title(title)
    plt.plot()
    plt.savefig(address+"/goal"+str(i)+'.png')
    plt.close()

monkey = "p"
day = 4
n_maps = 2
time = 26
start = 0
ep = 20
bias_lam = [2,4.0]
cost = 'mse' #'ce'
lr_maps = 0.001
end = start + ep - 1

address = 'irl/'+monkey+'/day'+str(day)+'/'+str(cost)+'_'+str(start)+'_'+\
        str(end)+'_map'+str(n_maps)+'_time'+str(time)+'map_lr'+str(lr_maps)+'_bias'+str(bias_lam)

if __name__ == "__main__" :

    #for i in range(time) :
        #make_dirl_heatmap(i, 3.4)


    for i in range(n_maps) :
        make_dirl_goal_map(i, 4.7)