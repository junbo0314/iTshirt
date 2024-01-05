import matplotlib.pyplot as plt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse, os
import pickle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from itertools import permutations
from src.compute_conf_interval import compute_conf_interval, compute_inv_hessian
from src.dirl_for_gridworld import fit_dirl_gridworld
from MonkeyPath import *
from env_monkey import *
from src.plot_and_save import *
import multiprocessing
from functools import partial 



def logging_time(original_fn) :
    def wrapper_fn(*args, **kwargs) :
        start_time = time.time()
        result = original_fn(*args, **kwargs) 
        end_time = time.time()
        print("Working Time[{}]: {} sec".format(result, end_time-start_time))
        return result
    return wrapper_fn




def get_gen_recovered_parameters(seed, n_maps, lr_weights, lr_maps, num_trajs, max_iters,
                                 trajectories, P_a, sigma,REC_DIR_NAME):
    """ makes a summary plot for recovered and generative parameters
        args:
            seed (int): which seed to plot
            n_maps (int): how many goal maps to pot
            lr_weights (float): which learning rate to plot
            lr_maps (float): which learning rate to plot
            save (bool): whether to save the plot or not
    """
    # directory for loading recovered and generative weights

    rec_dir_name = REC_DIR_NAME + "/maps_"+str(n_maps)+\
                            "_lr_"+str(lr_maps)+"_"+ str(lr_weights) + "/"

    # load recovered parameters for this seed
    rec_weights = np.load(rec_dir_name + "weights_trajs_" + str(num_trajs) +
                            "_seed_" + str(seed) + "_iters_" + str(max_iters) +
                            ".npy")[-1]
    # load recovered parameters for this seed
    rec_goal_maps = np.load(rec_dir_name + "goal_maps_trajs_" +
                                str(num_trajs) + "_seed_" + str(seed) +
                                "_iters_" + str(max_iters) + ".npy")[-1]

    # compute rewards
    rec_rewards = rec_weights.T @ rec_goal_maps



    return rec_rewards, rec_goal_maps, rec_weights



########################################################################
def fit_dirl(monkey, Monkey_Path, env, REC_DIR_NAME, n_maps,lr_weights, lr_maps, 
             max_iters, gamma, sigma, P_a, bias_goal_map, seed, window, i, df, day, start, 
             end, num_trajs, T=0, bias_lam=[0,0,0,0]) :
    trajectories, T = Monkey_Path.get_sa_reverse_pad(day, start, num_trajs,T,past=False)
    LL=fit_dirl_gridworld(n_maps, lr_weights, lr_maps, max_iters, gamma, seed, 
            trajectories, P_a, bias_goal_map, bias_lam, REC_DIR_NAME)
    
    df.iloc[window,i]=round(LL,6)
    print(window, i, LL)
    file_path = 'data/'+monkey+'/LLtable_'+str(bias_lam)+'.csv'
    df.to_csv(file_path, index=True, encoding='utf-8')            

    rec_rewards, rec_goal_maps, rec_weights  = get_gen_recovered_parameters(
seed, n_maps, lr_weights,lr_maps, num_trajs, max_iters, trajectories, P_a, sigma, REC_DIR_NAME)

    DIR_NAME = monkey+'/day'+str(day)+'/'+str(start)+'_'+\
    str(end)+'_map'+str(n_maps)+'_time'+str(T)+'_bias'+str(bias_lam)
    PAR_DIR_NAME = 'parameters/'+DIR_NAME

    np.save(PAR_DIR_NAME+'_weights.npy', rec_weights)
    
# save reward and goal_map into csv file
    reward = rec_rewards - np.min(rec_rewards) +0.1
    goal_maps = rec_goal_maps - np.min(rec_goal_maps) + 0.1

    for t in range(reward.shape[0]) :
        reward_map = env.transform_irl_reward(reward[t], 
                                DIR_NAME+'/','time'+str(t)+'.csv')

    for k in range(goal_maps.shape[0]) :
        goal_map = env.transform_irl_reward(goal_maps[k], 
                                DIR_NAME+'/','goal'+str(k)+'.csv')
        
    plot_final_weights(DIR_NAME, rec_weights, n_maps, T)
########################################################################

def fit_dirl_func(args) :
    monkey, Monkey_Path, env, lr_weights, lr_maps, max_iters, gamma, sigma, P_a, trial, n_maps, seed, df, day, bias_lam = args
    REC_DIR_NAME = 'data/'+monkey+'/day'+str(day)


    bias_goal_map = Monkey_Path.get_goal_map(day,trial, n_maps)
    func = partial(fit_dirl, monkey, Monkey_Path, env,REC_DIR_NAME, 
                        n_maps,lr_weights, lr_maps, max_iters, 
                        gamma, sigma, P_a, bias_goal_map, seed,
                         df, day, num_trajs)
    func(bias_lam)







if __name__=='__main__':
    
    TRAIN_DIRL_NOW = True

    grid_H, grid_W = 11, 11 # size of gridworld
    day_idx = {"p" : [2,57,122,197,263,328,398,463,519,645], 
               's' : [0,95,170,240,335,400,460,525,586,645]}
    # caution!!! monkey s's day1 ends in 91!!! 92,93,94 is not available.
    monkey = "s"
    Monkey_Path = MonkeyPath(monkey_name=monkey)
    env = Monkey()
    
    SAVE_DIR = 'figures/'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if TRAIN_DIRL_NOW:
        
        # parameters to reproduce Fig 3
        max_iters=100 # max iters to run SGD for optimization of goal maps and weights durng each outer loop of dirl
        seed = 1 # initialization seed
        gamma = 0.9 # discount factor
        # load parameters
        P_a = np.load('parameters/P_a.npy')
        sigma = 2**(-3.5)

        lr_maps = 0.001 # lr of goal maps
        lr_weights = 0.09 # lr of weights
    

        n_maps = 4 # goal maps
        bias_lam = [25,25,25,25]
        window_size = 3
        df = pd.DataFrame(0, index=range(10), columns=range(8))
        df = df.astype(float)
        

        args_list = [(monkey, Monkey_Path, env, lr_weights, lr_maps, max_iters, gamma, sigma, P_a, num_trajs, n_maps, seed, df, day, window, window_size, bias_lam)  for day in range(2, 10) for window in range(10)]

        with multiprocessing.Pool(processes=8) as pool :
            pool.map(fit_dirl_func, args_list)
  
                






'''

@logging_time
def process_A() :
    list(map(random_walk, range(20)))

process_A()

from multiprocessing import Pool
~`
@logging_time
def process_B() :
    num_cores = os.cpu_count()
    pool = Pool(num_cores)
    pool.map(random_walk, range(20))

process_B()

'''

