import torch
import pickle
import numpy as np 
import os 

"""For saving model's network's multilayer perceptron weights (given save rate (episode num))"""


def save_network_data(model, trial_num, save_rate=1000, dir="data/"):
    if trial_num % save_rate == 0: 
        dir_path = dir + "model_parameters/"
        if not(os.path.exists(dir_path)):
            os.makedirs(dir_path)

        file_name = dir_path + f"{str(trial_num)}.pt" 
        torch.save(model.state_dict(), file_name)
