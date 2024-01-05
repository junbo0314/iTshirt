clc
close all
clear all

load("data/P_OFC.mat")
load("data/reward_pos_trial.mat")


pos_sequence_all = cell(size(bhv,1) ,1);


parfor trial_num = 1:size(bhv,1)

pos_data = bhv{trial_num}(1100:end,1:2);

if ~strcmp(current_pos(pos_data(1,:)), 'Z')
    pos_all = {char(current_pos(pos_data(1,:)))};
else
    pos_all = {};
end


for n = 2:size(pos_data,1)
    pos = char(current_pos(pos_data(n,:)));
    if strcmp(pos , 'Z')
        continue
    end
    if isempty(pos_all)
        pos_all{1} = pos;

    elseif ~strcmp(pos , pos_all{end})
        pos_all{end+1} = pos;
    end
   
end

pos_end = pos_all{end};
rewarded_pos = reward_position(reward_pos_all{trial_num});

if strcmp(pos_end , rewarded_pos{1})
    pos_all{end+1} = rewarded_pos{2};


else
    pos_all{end+1} = rewarded_pos{1};

end


pos_sequence_all{trial_num} = pos_all;

end


save("data/Position_sequence_P" , "pos_sequence_all");