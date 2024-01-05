clc
clear all
close all 



load("./data/pos_seq_s.mat")

a = 0; 
while a ==0

for trial_num = 1:519
    if trial_num == 1
        a=1;
    end
    i = 0;
    trial_path = pos_sequence_all{trial_num};
    for state_num = 1:size(trial_path,1) -1

        state = trial_path(state_num, :);
        state_1 = trial_path(state_num+1, :); 

        vec = state_1 - state;
        if vec(1)^2 + vec(2)^2 == 2
            a= 0; 
            
            state_between = BetweenState(state, state_1);
            trial_path = [trial_path(1:state_num,:); state_between; trial_path(state_num+1:end,:)];

            break



        end


    end

    pos_sequence_all{trial_num} = trial_path;
    if not(a==1) && not(trial_num==519)
        a = 0; 


    end


end
end

save("pos_seq_s.mat", "pos_sequence_all")

function state_between = BetweenState(state_1, state_2) 
map =  vertcat([2, 0], [4, 0], [8, 0], [2, 1], [4, 1], [8, 1], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [2, 3], [4, 3], [6, 3], [8, 3], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4], [10, 4], [2, 5], [4, 5], [6, 5], [8, 5], [0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6], [8, 6], [4, 7], [6, 7], [8, 7], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [8, 8], [6, 9], [6, 10]);

transition_vec = state_2 - state_1; 
possible_state_1 = state_1 + [transition_vec(1), 0];
possible_state_2 = state_1 + [0,transition_vec(2)]; 

if ismember(possible_state_1, map, "rows")
    state_between = possible_state_1;

elseif ismember(possible_state_2, map, "rows")
    state_between = possible_state_2; 
else
    error("error! value is wrong")

end

end