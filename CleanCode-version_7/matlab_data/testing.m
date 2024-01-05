addpath("analysis_tool")
addpath("data")
addpath("plotting_tool")
clear all
close all 
show_map 
load("P_OFC.mat")

load("pos_seq_p.mat")
%{

a = [];


for i = 1:size(bhv,1)
    trial_end = bhv{i}(end, 1:2); 
    goal_position = bhv{i}(end, 9:10); 
    vec = trial_end - goal_position;
    diff = sqrt(vec(1)^2 + vec(2)^2);
    a = [a; diff]; 
end


a



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
grid = [];

for trial_num = 1:10


trial_data = bhv{trial_num}(:, 1:2);

for i = 1000:size(trial_data, 1)
    curr_pos = pos2grid(trial_data(i, 1:2));
    grid = [grid ; curr_pos ];


end



end


unique_grid = unique(grid, "rows");



function output = pos2grid(position)

output = round(position/7 + 5); 

end


%}

a = [];
for i = 1:size(pos_sequence_all,1)
    trial_end = pos_sequence_all{i}(end, :)
    a = [a; trial_end];

end



function output = pos2grid(position)

output = round(position/7 + 5); 

end

