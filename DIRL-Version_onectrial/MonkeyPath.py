import numpy as np
import scipy.io
import random
import pickle


class MonkeyPath:
    """ "For getting Monkey grid positional data, use get_~ to get the start of goal position of given trial_num
    and use get_trial for sequence of certain trial"""

    random.seed(18)


    def __init__(self, monkey_name="p"):
        mat_file_name = f"matlab_data/pos_seq_{monkey_name}"
        mat_file = scipy.io.loadmat(mat_file_name)
        pos_sequence = mat_file["pos_sequence_all"]
        self.pos_sequence = pos_sequence

        day_idx = {"p" : [2,57,122,197,263,328,398,463,519], 
               's' : [0,95,170,240,335,400,460,525,586,645]}
        self.date = day_idx[monkey_name]

        self.new_pos = []
        for day in range(len(self.date)-1) :
            one_day = pos_sequence[self.date[day]:self.date[day+1]]
            one_day_without_outlier = self.outlier(one_day)
            self.new_pos.append(one_day_without_outlier)
            

        self.trial_num = self.pos_sequence.shape[0]
        self.location_num = 16

        start_end_location = []
        for i in range(self.trial_num):
            start_end_location.append(tuple(self.pos_sequence[i][0][0]))

        start_end_location = list(set(start_end_location))
        self.start_end_locations = start_end_location

        
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

        marking_p = [[0,0,0,0,0],
               [2,15,29,43,57],
               [57,73,89,105,122],
               [122,140,169,178,197],
               [197,213,229,246,263],
               [263,279,295,311,328],
               [328,345,362,380,398],
               [398,414,430,446,463],
               [463,477,491,505,519]]

        marking_s = [[0,0,0,0,0],
                [0,23,46,69,92],
                [95,113,132,151,170],
                [170,187,204,222,240],
                [240,263,287,311,335],
                [335,351,367,383,400],
                [400,415,430,445,460],
                [460,476,492,508,525],
                [525,540,555,570,586],
                [586,600,615,630,645]]
        

    def get_goal_position(self, trial_num):
        return self.pos_sequence[trial_num][0][-1]

    def get_start_position(self, trial_num):
        return self.pos_sequence[trial_num][0][0]

    def get_trial(self, trial_num):
        return self.pos_sequence[trial_num][0]

    def start_locations(self):
        starting_locations = []
        for i in range(self.trial_num):
            starting_locations.append(tuple(self.pos_sequence[i][0][0]))
        #starting_locations = set(starting_locations)
        #starting_locations = list(starting_locations)

        return starting_locations

    def end_locations(self):
        end_locations = []
        for i in range(self.trial_num):
            end_locations.append(tuple(self.pos_sequence[i][0][-1]))

        #end_locations = list(set(end_locations))
        return end_locations

    def location2index(self, location):
        index_num = self.start_end_locations.index(tuple(location))
        return index_num


    def get_sas(self, day, start_trial, size) :
        
        trials = self.new_pos[day-1][start_trial:start_trial+size]
        trials = trials.squeeze()
        data_trajectories = []
        for trial in trials :
            sas = []
            for t in range(len(trial)-1) :
                action = 0
                for a in range(4) :
                    if ((trial[t] + self.action[a]) == trial[t+1]).all() :
                        action = a
                sas.append([trial[t], action, trial[t+1]])
            data_trajectories.append(sas)

        return data_trajectories
    

    def get_one_trial(self, day, trial) :
        '''
        day: the day data
        trial: the trial of the day
        ex) day=2, trial=0 day2's first trial trajectory data
        '''
        trials = self.new_pos[day-1][trial]
    

        trajectories = []
        for trial in trials : #한 트레젝토리의 일련의 과정이 나옴, 방문 ''state''
            presa = [] #state와 action을 위한 칸(초기화)

            for t in range(len(trial)-1) : #trial의 길이는 방문 state의 개수
                action = 0
                for a in range(5) :
                    if ((trial[t] + self.action[a]) == trial[t+1]).all() :
                        action = a
                s = self.to_index[tuple(trial[t])]
                presa.append([s, action]) #presa에는 trajectory의 모든 s와 a가 들어있음
            trajectories.append(presa)
        return trajectories


    def get_sa_reverse_pad(self, day, start_trial, size, T, past=False) :
        '''
        past : True면 start_trial을 기준으로 거꾸로 size만큼 trial을 뽑아냄
                ex) start trial - size ~ start trial 까지
        '''
        if past :
            if start_trial >= size :
                trials = self.new_pos[day-1][start_trial-size:start_trial]
            else :
                trials = np.concatenate((self.new_pos[day-2][-(size-start_trial):],
                                          self.new_pos[day-1][:start_trial]), axis=0)
        else :
            trials = self.new_pos[day-1][start_trial:start_trial+size]
        trials = trials.squeeze()
        
        if T == 0 :
            # 각 리스트의 길이를 계산하여 리스트에 저장
            lengths = [len(trial) for trial in trials]
            T = int(sum(lengths) / len(lengths))

        trajectories = []
        for trial in trials : #한 트레젝토리의 일련의 과정이 나옴, 방문 ''state''
            presa = [] #state와 action을 위한 칸(초기화)

            for t in range(len(trial)-1) : #trial의 길이는 방문 state의 개수
                action = 0
                for a in range(5) :
                    if ((trial[t] + self.action[a]) == trial[t+1]).all() :
                        action = a
                s = self.to_index[tuple(trial[t])]
                presa.append([s, action]) #presa에는 trajectory의 모든 s와 a가 들어있음

            if len(presa) < T :
                sa=presa[::-1] #sa는 presa의 역전 형태임
                for j in range(len(presa)+1,T+1) : #부족한 차이만큼
                    sa.append([self.to_index[tuple(trial[0])], 4]) 
                sa=sa[::-1]              
                trajectories.append(sa)

            elif len(presa) >= T :    
                sa=presa[::-1]
                sa=sa[:T]
                sa=sa[::-1]
                trajectories.append(sa)

        return trajectories, T
        


    def get_goal_map(self, day, trial, n_maps) :
        '''
        input
            - start : the order of trial in the given day
            start 트라이얼로부터 직전 20개 트라이얼을 찾아내야하는게관건
        '''

        nj_loc = self.new_pos[day-1][0][0][-1]
        oj_loc = self.new_pos[day-2][0][0][-1]

        u = np.zeros((n_maps,52))
        
        # new_jackpot
        u[0] = self.smooth_reward(nj_loc, u[0])
        
        # old_jackpot
        u[1] = self.smooth_reward(oj_loc, u[1])

        if n_maps >= 3 :
            # subgoal
            subgoals=[38,2,13,3,27,15,36]
            for i in subgoals :
                u[2][i] = 1

        # Exploration map
        if n_maps >= 4 :
            # 방문횟수 채워넣을 np
            visit = np.zeros((52))
            # find visit frequency for latest 20 trials

            trajectories = self.get_one_trial(day,trial)
            for t in range(20) :
                # 여기서부터 어디 들렷는지 횟수 계산
                for state, action in trajectories[t] :
                    visit[state] += 1
            u[3] = 4 - np.log(visit + 1)
        return u
    
    
    
    def smooth_reward(self, jackpot, goal_map) :
        '''
        jackpot : (np.array) 1x2
        goal_map : (np.array) 1 x num_states
        max_reward : (int) reward of the jackpot

        function :
            - fill the states with smooth reward , relative to distance
              using softmax
        '''

        num_states = goal_map.size
        for s in range(num_states) :
            state = self.to_state[s]
            distance = abs(state[0]-jackpot[0])+abs(state[1]-jackpot[1])
            goal_map[s] = -distance
        goal_map = np.exp(goal_map)/np.sum(np.exp(goal_map))
        goal_map= (goal_map-np.min(goal_map)/(np.max(goal_map)-np.min(goal_map)))

        return goal_map


    def outlier(self, data):
        data = data.squeeze()
        trial_length={}
        for trial_idx, trial_data in enumerate(data):
                trial_length[trial_idx] = len(trial_data)
        sorted_dict = dict(sorted(trial_length.items(), key=lambda item: item[1]))
        total_items=len(sorted_dict)
        exclude_count=int(0.2 *total_items)
        remaining_items=list(sorted_dict.items())[exclude_count:-exclude_count]
        sorted_list = sorted(remaining_items, key=lambda x: x[0])
        survive_trial = [item[0] for item in sorted_list]
        data=data[survive_trial]
        data=np.array(data)
        data = data.reshape((len(data), 1))
        return data

    





if __name__ == "__main__":
    monkey_path = MonkeyPath()
    print(len(monkey_path.new_pos))
