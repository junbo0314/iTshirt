import numpy as np 


def exploration_entropy(path):
    state_counter = {}
    total_cell_num = 52 
    for state in path:
        if not(tuple(state) in state_counter):
            state_counter[tuple(state)] = 0
        state_counter[tuple(state)] += 1

    state_counter = np.array(list(state_counter.values()))
    state_p = state_counter/sum(state_counter)
    entropy = -sum(state_p*np.log(state_p))

    visited_cell_num = len(state_counter)
    explore_map_percentage = visited_cell_num/total_cell_num * 100 

    return entropy, explore_map_percentage


if __name__ == "__main__":
    import pickle

    with open("example_trajectories", "rb") as f:
        example_path = pickle.load(f)
    print(example_path)

    entropy = exploration_entropy(example_path)
    print(entropy)
