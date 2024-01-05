from .choice_sim_percentage import choice_sim_percent
from .monkey_max_choice_agent_compare import monkey_max_choice_agent_compare
from .exploration_entropy import exploration_entropy
from .path_similarity import path_similarity 
from .shannon_info import shannon_info
import numpy as np 
import sys, os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from agent_package.ModelBasedAgent import ModelBasedAgent 
from torch.utils.tensorboard import SummaryWriter
from .DataSaver import DataSaver 

def get_all_path_analytic_out(agent = ModelBasedAgent, agent_path = np.array , agent_action_seq = np.array, monkey_path = np.array,
                              writer = SummaryWriter, trial_num = int, total_reward = float, total_length = int,
                              data_saver = DataSaver):
    
    choice_sim_percent_monkey_agent = choice_sim_percent(agent, monkey_path)
    choice_sim_percent_agent_self = choice_sim_percent(agent, agent_path) 

    choice_sim_percent_monkey_agent_max = monkey_max_choice_agent_compare(monkey_path, agent)

    path_similarity_monkey_agent = path_similarity(monkey_path, agent_path) 

    agent_exploration_entropy, agent_explore_percent = exploration_entropy(agent_path)

    shannon_info_value = shannon_info(agent_path, agent_action_seq)

    path_analytic_out_dict = {"choice_sim_percent_monkey_agent": choice_sim_percent_monkey_agent,
                              "choice_sim_percent_agnet_self": choice_sim_percent_agent_self,
                              "choice_sim_percent_monkey_agent_max": choice_sim_percent_monkey_agent_max,
                              "path_similarity_monkey_agent": path_similarity_monkey_agent,
                              "agent_exploration_entropy": agent_exploration_entropy,
                              "agent_expore_percent": agent_explore_percent,
                              "shannon_info_value": shannon_info_value,
                              }
    
    writer.add_scalar("reward", total_reward, trial_num) 
    writer.add_scalar("length", total_length, trial_num)
    writer.add_scalar("reward_rate", total_reward/total_length, trial_num)
    writer.add_scalar("epsilion", agent.epsilon, trial_num)
    writer.add_scalar("shannon_info_value:", shannon_info_value, trial_num)
    writer.add_scalar("path_sim", path_similarity_monkey_agent, trial_num) 
    writer.add_scalar("choice_sim_percentage" , choice_sim_percent_monkey_agent, trial_num) 
    writer.add_scalar("monkey_max_choice_agent_compare", choice_sim_percent_monkey_agent_max, trial_num)
    writer.add_scalar("exploration_entropy", agent_exploration_entropy, trial_num)
    writer.add_scalar("exploration_percentage", agent_explore_percent, trial_num)
    writer.add_scalar("choice_sim_percent_agnet_self", choice_sim_percent_agent_self, trial_num)

    data_saver.record_data(length=total_length, reward=total_reward, reward_rate=total_reward/total_length,
                           shannon_value=shannon_info_value, path_sim=path_similarity_monkey_agent,
                           choice_sim_percent=choice_sim_percent_monkey_agent,
                           monkey_max_choice_compare_percent=choice_sim_percent_monkey_agent_max,
                           exploration_entropy=agent_exploration_entropy,
                           explore_percentage=agent_explore_percent,
                           trajectory=agent_path,
                           choice_sim_percent_agent_self=choice_sim_percent_agent_self)


    




if __name__ == "__main__":
    import pickle
    import numpy as np
    import sys, os 
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from MonkeyPath import MonkeyPath 

    agent = ModelBasedAgent(2,4,256,0.0001,10000,64,0.99,explore_agent=True)
    monkey_path = MonkeyPath() 
    agent_path_ex = monkey_path.get_trial(100)
    monkey_path_ex = monkey_path.get_trial(101)  

    monkey_path_len = len(agent_path_ex)
    action_seq_ex = np.random.random_integers(0, 3, monkey_path_len)

    with open("example_trajectories", "rb") as f:
        example_path = pickle.load(f)

    get_all_path_analytic_out(agent=agent, agent_path=agent_path_ex, agent_action_seq=action_seq_ex, monkey_path=monkey_path_ex)
   