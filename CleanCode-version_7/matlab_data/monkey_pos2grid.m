clc
clear all 
close all 
addpath("analysis_tool/")
addpath("data/") 
load('S_OFC.mat')


%{
Monkey bhv data consists of trial(519) * time
 each data point is single time point with 10 element 
      
    Position    Joystick movement    Eyelink      Starting position    Goal position  
[(x_pos, y_pos), (x_mov, y_mov), (x_eye, y_eye), (x_start, y_start), (x_goal , y_goal)]

note
1) position data starts at ~1000ms to 1100ms (trash data ahead) 
2) idk about joystick data 
3) Eyelink data is trash 



%}



pos_sequence_all = cell(size(bhv,1) ,1);
map = [[2, 0]; [4, 0]; [8, 0];
            [2, 1]; [4, 1]; [8, 1];
            [0, 2]; [1, 2]; [2, 2]; [3, 2]; [4, 2]; [5, 2]; [6, 2]; [7, 2]; [8, 2];
            [2, 3]; [4, 3]; [6, 3]; [8, 3];
            [2, 4]; [3, 4]; [4, 4]; [5, 4]; [6, 4]; [7, 4]; [8, 4]; [9, 4]; [10, 4];
            [2, 5]; [4, 5]; [6, 5]; [8, 5];
            [0, 6]; [1, 6]; [2, 6]; [3, 6]; [4, 6]; [5, 6]; [6, 6]; [7, 6]; [8, 6];
            [4, 7]; [6, 7]; [8, 7];
            [2, 8]; [3, 8]; [4, 8]; [5, 8]; [6, 8]; [8, 8];
            [6, 9];
            [6, 10]];

parfor trial_num = 1:size(bhv,1)


pos_data = bhv{trial_num}(1000:end,1:2);

start_pos = bhv{trial_num}(end, 7:8);
goal_pos = bhv{trial_num}(end, 9:10); 

start_time = 1;
for i = 1:100  %% calculate the starting time of trial using sim_point(distance <0.5) 
    start_time = i;
    if sim_point(start_pos, pos_data(i, 1:2))

        break 
    end

end

pos_data = pos_data(i:end, :);

curr_pos = pos2grid(pos_data(1,:))
pos_seq = []; 

for time = 1:size(pos_data, 1)
    if ~ismember(pos2grid(pos_data(time, :)), map, 'rows')
        continue
    end
    past_pos = curr_pos 
    curr_pos = pos2grid(pos_data(time, :));

    if isempty(pos_seq)
        pos_seq = curr_pos;
    end

    if ~all(curr_pos == past_pos)
        pos_seq = [pos_seq; curr_pos];


    end

    if sim_point(goal_pos, pos_data(time,:))
        break 

    end
     
    


end

if not(sum(pos_seq(end,:) == pos2grid(bhv{trial_num}(end, 9:10))) == 2)
    pos_seq = [pos_seq; pos2grid(bhv{trial_num}(end, 9:10))];
end

pos_sequence_all{trial_num} = pos_seq



end


save("data/pos_seq_s.mat", "pos_sequence_all");
fit_missing_points


function output = pos2grid(position)

output = round(position/7 + 5); 

end


function sim = sim_point(a,b)

vector = a-b;
vec_len = sqrt(vector(1)^2 + vector(2)^2);
if vec_len< 1.5 
    sim = 1;
   
else
    sim = 0;

end

end


