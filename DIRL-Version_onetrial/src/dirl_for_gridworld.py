import argparse, os
import numpy as np
import pickle
from src.optimize_weights import getMAP_weights
from src.optimize_goal_maps import getMAP_goalmaps, neglogll
from src.compute_validation_ll import get_validation_ll


def fit_dirl_gridworld(N_MAPS,trial,lr_weights, lr_maps, max_iters, gamma, seed,
                         trajectories, P_a, bias_goal_map,inherit_maps,lam, 
                         REC_DIR_NAME) :
    """ fits DIRL on simulated trajectories from the gridworld environment given hyperparameters
        and saves all recovered parameters

        args:
        version (int): choose which version of the simulated trajectories to use
        num_trajs (int): choose how many trajectories to use
        lr_weights (float): choose learning rate for weights
        lr_maps (float): choose learning rate for goal maps
        max_iters (int): num iterations to run the optimization for weights/goal maps per outer loop of dirl
        gamma (float): value iteration discount parameter
        N_MAPS (int): # of goal maps to use while fitting DIRL
        seed (int): initialization seed
        GEN_DIR_NAME (str): name of the folder that contains the trajectories and generative parameters
        REC_DIR_NAME (str): name of the folder to store recovered parameters
    """
    np.random.seed(seed)

    # create folder to store recovered parameters

    save_dir = REC_DIR_NAME + "/maps_"+str(N_MAPS)+ "_bias" + str(lam)

    # check if save_dir exists, else create it 
    if not os.path.isdir(save_dir): 
        os.makedirs(save_dir, exist_ok = True)

    T = len(trajectories[0])
    # split into train and val sets


    # loading some relevant generative parameters

    N_STATES = P_a.shape[0] # no of states in gridworld
    sigma = 2**(-3.5) # noise covariance of time-varying weights
    sigmas = [sigma]* N_MAPS

    # choose a random initial setting for the weights (parameters)
    weights = (np.random.multivariate_normal(mean=np.zeros(T,), cov = sigmas[0]*np.eye(T,), size=N_MAPS)).reshape((N_MAPS,T))
    # choose a random initial setting for the goal maps (parameters)
    goal_maps = np.random.uniform(size=(N_MAPS,N_STATES))

    # save things
    rec_weights = []
    rec_goal_maps = []
    losses_all_weights = []
    losses_all_maps = []
    val_lls = []

    for i in range(20):
        #print("At iteration: "+str(i), flush=True)
        #print("-------------------------------------------------", flush=True)
        # get the MAP estimates of time-varying weights and list of losses at every time step
        a_MAPs, losses =  getMAP_weights(seed, P_a, trajectories, hyperparams = sigmas, goal_maps = goal_maps, 
                                                        a_init=weights, max_iters=max_iters, lr=lr_weights, gamma=gamma)
        weights = a_MAPs[-1]
        rec_weights.append(weights)
        losses_all_weights = losses_all_weights + losses

        # save recovered time-varying weights as well as training loss
        np.save(save_dir + "/weights_trial_" + str(trial) +".npy", rec_weights)
        np.save(save_dir + "/losses_weights_trajs_"+str(trial)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", losses_all_weights)

        # get the optimal estimates of the goal maps and list of losses at every time step
        goal_maps_MLEs, losses =  getMAP_goalmaps(seed, P_a, bias_goal_map,inherit_maps, trajectories, hyperparams = sigmas, a=weights, 
                                                        goal_maps_init = goal_maps, max_iters=max_iters, lr=lr_maps,
                                                        gamma=gamma, bias_lam=lam)

        goal_maps = goal_maps_MLEs[-1]
        rec_goal_maps.append(goal_maps)
        losses_all_maps = losses_all_maps + losses

        # save recovered goal maps as well as training loss
        # 'data/'+monkey+'/day'+str(day)
        np.save(save_dir + "/goal_maps_trial_" + str(trial) +".npy", rec_goal_maps)
        np.save(save_dir + "/losses_maps_trajs_"+str(trial)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", losses_all_maps)

        val_ll = get_validation_ll(seed, P_a, trajectories, hyperparams = sigmas, a=weights, goal_maps=goal_maps, gamma=gamma)
        val_lls.append(val_ll)
        # save validation LL on held-out trajectories
        np.save(save_dir + "/validation_lls_"+str(trial)+".npy", val_lls) 

    LL = (val_lls[-1])/ (len(trajectories)*T) / np.log(2)
    
    return LL