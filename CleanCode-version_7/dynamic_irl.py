import numpy as np
import pandas as pd
from env_monkey import *


monkey = "p"
day = 2
n_maps = 2
time = 22
start = 0
ep = 39
bias_lam = [1,1]
cost='mse'

end = start + ep - 1



address = 'parameters/'+monkey+'/day'+str(day)+'/'+str(start)+'_'+\
        str(end)+'_map'+str(n_maps)+'_time'+str(time)+'_bias'+str(bias_lam)+'_'+str(cost)

reward = np.load(address+'_reward.npy')
goal_maps = np.load(address+'_goal_maps.npy')


env = Monkey()

print(reward.shape)

reward = reward - np.min(reward) +0.1

for t in range(reward.shape[0]) :
    reward_map = env.transform_irl_reward(reward[t], 
                                          monkey+'/day'+str(day)+'/'+str(cost)+'_'+str(start)+'_'+\
        str(end)+'_map'+str(n_maps)+'_'+str(time)+'_bias'+str(bias_lam)+'/','time'+str(t)+'.csv')


goal_maps = goal_maps - np.min(goal_maps) + 0.1

for k in range(goal_maps.shape[0]) :
    goal_map = env.transform_irl_reward(goal_maps[k], 
                                        monkey+'/day'+str(day)+'/'+str(cost)+'_'+str(start)+'_'+\
        str(end)+'_map'+str(n_maps)+'_'+str(time)+'_bias'+str(bias_lam)+'/','goal'+str(k)+'.csv')