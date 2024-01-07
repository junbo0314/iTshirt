import numpy as np
import pandas as pd
import os

class Monkey():
    def __init__(self) :
        
        self.map = np.array([
                [2, 0],[4, 0],[8, 0],
                [2, 1],[4, 1],[8, 1],[0, 2],[1, 2],[2, 2],[3, 2],[4, 2],[5, 2],[6, 2],[7, 2],[8, 2],
                [2, 3], [4, 3],
                [6, 3],[8, 3],
                [2, 4],[3, 4],[4, 4],[5, 4],[6, 4],[7, 4],[8, 4],[9, 4],[10, 4],
                [2, 5],[4, 5],[6, 5],[8, 5],
                [0, 6],[1, 6],[2, 6],[3, 6],[4, 6],[5, 6],[6, 6],
                [7, 6],[8, 6],
                [4, 7],[6, 7],[8, 7],
                [2, 8],[3, 8],[4, 8],[5, 8],[6, 8],[8, 8],
                [6, 9],
                [6, 10],
            ])

        self.action = {
                        0: np.array([1,0], dtype=np.float16), 
                        1: np.array([0,1], dtype=np.float16),
                        2: np.array([-1,0], dtype=np.float16),
                        3: np.array([0,-1], dtype=np.float16),
                        4: np.array([0,0], dtype=np.float16)
                      }
        
        self.n_states = self.map.shape[0]
        self.n_actions = 5
        
        self.to_index = {}
        idx = 0
        for i in range(11):
            for j in range(11):
                if [i, j] in self.map.tolist() :
                    self.to_index[(i,j)] = idx
                    idx += 1
                    
        # key : index , value : array
        self.to_state = {v:k for k,v in self.to_index.items()}
        
        
    def calculate_transition_probability(self) :
        
        transition_p = np.zeros((self.n_states, self.n_states, self.n_actions))
        
        for s in range(self.n_states) :
            for a in range(self.n_actions) :
                next_state = np.array(self.to_state[s]) + self.action[a]
                if next_state.tolist() in self.map.tolist() :
                    next_state_index = self.to_index[tuple(next_state)]
                    transition_p[s,next_state_index,a] = 1

        #np.save('/Users/swankim/Desktop/Excool_Assignment/CleanCode-version_2/parameters/P_a.npy', 
        #        transition_p)

        '''
        transition_count = np.zeros((self.n_states, self.n_actions, self.n_states))

        for trajectory in trajectories:
            for state,action,next_state in trajectory :
                transition_count[state,action,next_state] += 1
           
        for s in range(self.n_states) :
            for a in range(self.n_actions) :
                total = transition_count[s,a,:].sum()
                if total != 0 :
                    transition_count[s,a,:] /= total
        '''             
        
        return transition_p
    
    
    
    def transform_trajectories(self, trajectories) :
        
        trajs = []
        for trajectory in trajectories :
            traj = []
            for state, action, next_state in trajectory :
                state = tuple(state)
                next_state = tuple(next_state)
                
                traj.append([self.to_index[state],action,self.to_index[next_state]])
                
            trajs.append(traj)
        
        return trajs

                
                
    def transform_irl_reward(self, reward, address, name) :
        
        reward_map = pd.DataFrame(0.0, index=range(11), columns=range(11))
        
        for i in range(self.n_states) :
            state = self.to_state[i]
            
            reward_map.iloc[state[0], state[1]] = round(reward[i], 5)
        
        save = 'irl/'+ address
        if not os.path.exists(save):
            os.makedirs(save)

        reward_map.to_csv(save+name, index=False)

                
    def save_svf(self, svf) :
        
        svf_map = pd.DataFrame(0, index=range(11), columns=range(11))
        
        for i in range(self.n_states) :
            state = self.to_state[i]
            
            svf_map.iloc[state[0], state[1]] = svf[i]
            
        svf_map.to_csv('/Users/swankim/Desktop/Excool_Assignment/CleanCode-version_2/irl/20ep/svf.csv')



if __name__ == "__main__" :
    env = Monkey()
    print(env.to_index[(7,2)])
    print(env.to_index[(1,2)])
    print(env.to_index[(3,4)])
    print(env.to_index[(1,6)])
    print(env.to_index[(5,6)])
    print(env.to_index[(3,8)])
    print(env.to_index[(6,9)])
    print(env.to_state[32])
    #env.calculate_transition_probability()