clear all 
close all 

load("pos_seq_p.mat")
end_point_all = []; 

for i = 1:size(pos_sequence_all,1)
    trial_data = pos_sequence_all{i};
    end_point = trial_data(end, 1:2);
    end_point_all = [end_point_all ; end_point];

end
