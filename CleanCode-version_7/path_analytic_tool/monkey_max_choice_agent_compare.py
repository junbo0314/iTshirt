import numpy as np
from MonkeyPath import MonkeyPath
import sys, os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from agent_package.ModelBasedAgent import ModelBasedAgent


""" input trial_seq of monkey and agent, outputs similarity of max choice of monkey given state and agent's max_Q action"""


def monkey_max_choice_agent_compare(trial_seq = None, agent = ModelBasedAgent(1,1,1,1,1,1,1) ):
    direction_to_action = {
                    (1, 0): 0,
                    (0, 1): 1,
                    (-1, 0): 2, 
                    (0, -1): 3,
    }

    max_state_action = {} 

    for i, state in enumerate(trial_seq): 
        if i == len(trial_seq) - 1:
            break 

        direction_vec = np.array(trial_seq[i+1], dtype=np.float64) - np.array(state, dtype = np.float64)

        action = direction_to_action[tuple(direction_vec)]
        max_state_action.setdefault(tuple(state), [0, 0, 0, 0])
        max_state_action[tuple(state)][action] += 1 

    max_action_match_counter = 0 

    for state in max_state_action.keys(): 
        monkey_state_actions = max_state_action[state] 
        monkey_action = monkey_state_actions.index(max(monkey_state_actions))
        agent_action = agent.act(np.array(state), max_option = True)

        if monkey_action == agent_action:
            max_action_match_counter += 1 

    return 100*(max_action_match_counter/len(max_state_action))
