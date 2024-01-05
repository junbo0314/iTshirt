import numpy as np 
from MonkeyPath import MonkeyPath 



def min_length(point, path):
    point = np.array(point)
    path = np.array(path)
    min_len = 100
    for coordinate in path: 
        length = np.linalg.norm(point - coordinate)
        min_len = min(min_len, length)

    return min_len 



def path_similarity(seq_1 = 9, seq_2 = 21 ):
    monkey_path = MonkeyPath() 
    for num, seq in enumerate([seq_1, seq_2]): 
        if type(seq) == int: 
            if num == 1: 
                seq_1 = monkey_path.get_trial(seq_1)
            else: 
                seq_2 = monkey_path.get_trial(seq_2) 


    ##if not(seq_1[0] == seq_2[0]).all():
      ## raise ValueError("starting points are different!")
    
    max_len = 0 

    for point in seq_1:
        point_min_length = min_length(point, seq_2)
        max_len = max(max_len, point_min_length)



    return max_len 


def is_same_trial(start_location, end_location): 
    monkey_path = MonkeyPath() 
    index_list = [] 
    starting_locations = monkey_path.start_locations()
    end_locations = monkey_path.end_locations()
    for i in range(monkey_path.trial_num):
        starting_pos = monkey_path.get_start_position(i)
        ending_pos = monkey_path.get_goal_position(i) 
        if (starting_pos == np.array(starting_locations[start_location])).all() and (ending_pos == np.array(end_locations[end_location])).all():
            index_list.append(i)


    return index_list  
    

def same_trial_list():
    trial_index_list = [] 
    for i in range(16):
        for j in range(16):
            trial_list = is_same_trial(i, j)
            if not(len(trial_list) == 0): 
                trial_index_list.append(trial_list) 
    return trial_index_list 










    



