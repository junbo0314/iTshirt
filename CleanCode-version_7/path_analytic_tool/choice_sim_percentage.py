import numpy as np
import numpy as np
import sys, os 
import pickle
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from agent_package.ModelBasedAgent import ModelBasedAgent
from MonkeyMazeEnv import MonkeyMazeEnv
from path_analytic_tool.path_similarity import *


def choice_sim_percent(agent =ModelBasedAgent, path=[]):
    max_t = len(path)
    action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            }
    if max_t == 1:
        return 100 
    state = path[0]
    count = 0

    for t in range(max_t-1):
        state = path[t]
        next_agent_state = action_to_direction[agent.act(state, max_option=True)] + state
        next_monkey_state = path[t+1]
        if (next_agent_state == next_monkey_state).all():
            count += 1 

    simularity = 100*(count/(max_t-1))
    return simularity