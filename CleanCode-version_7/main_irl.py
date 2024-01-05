import numpy as np

from max_entropy import *
from env_monkey import *
from MonkeyPath import *


def main() :

    n_states = 52
    n_actions = 4
    
    feature_matrix = np.eye((n_states))
    
    env = Monkey()
    path = MonkeyPath()

    learning_rate = 0.01
    discount = 0.9

    num_tj = 30
    epochs = 300
    date = 1

    day = {1:2, 2:57, 3:122, 4:197, 5:263, 6:328}
    stop = {1:25, 2:32, 3:35, 4:9, 5:20}
    

    data_trajectories = path.get_sas(num_tj, day[date], day[date+1])
    trajectories = env.transform_trajectories(data_trajectories)

    transition_probability = env.calculate_transition_probability()
    
    irl_reward = irl(feature_matrix, n_actions, discount, transition_probability,
                     trajectories, epochs, learning_rate, stop[date])


    name = str(num_tj)+"ep/day"+str(date)+"/reward_"+str(epochs)+"_21.csv"
    env.transform_irl_reward(irl_reward, name)
    
if __name__ == "__main__" :
    main()
