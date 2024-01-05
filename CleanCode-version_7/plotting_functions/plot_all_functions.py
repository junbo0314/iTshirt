from .single_q_arrow_show import single_q_arrow_show 
from .model_q_arrow_show import model_q_arrow_show 
from .q_values_map import q_values_map 
import numpy as np 
import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from agent_package.ModelBasedAgent import ModelBasedAgent as Agent 
from agent_package.CuriosityCounterModelTable import CuriosityCounterModelTable as Model 


def plot_all_functions(agent=Agent, model=Model, i_episode=int, trial_t=int, state=np.array, reward_location = np.array,
                       data_dir = "plotting_functions_data/", dual_policy = False):


    ## for plotting with certain freqeuncy of trial_t 
    

    if not(trial_t == 100000000):
        return
    
    if dual_policy:
        model_q_arrow_show(agent=agent, model=model, i_episode=i_episode, counter=trial_t, state=state, data_dir=data_dir)
    else:
        single_q_arrow_show(agent=agent, i_episode=i_episode, counter=trial_t, state=state,reward_location=reward_location, data_dir=data_dir) 
    
    q_values_map(agent=agent, i_episode=i_episode, counter=trial_t, state=state, reward_location=reward_location, data_dir=data_dir)

