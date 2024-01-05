clc
clear all 
close all

load("data/Position_sequence_P.mat")
load("data/reward_pos_trial.mat")
trial_score_all = zeros(1, length(reward_pos_all));

parfor trial_num = 1:length(pos_sequence_all)

    trial_score = 0;
    single_trial = pos_sequence_all{trial_num};
    for time_num = 1:length(single_trial)
        if time_num == length(single_trial)
            trial_score = trial_score + 180;
            continue
        end

        eval = node_evaluate(single_trial{time_num} , reward_pos_all{trial_num});
        selected_index = find(char(single_trial{time_num+1}) == eval.index);

        trial_score = trial_score + eval.score(selected_index)





    end

    trial_score = trial_score/(length(single_trial));
    trial_score_all(trial_num) = trial_score;




end

change_day = zeros(1, 9);
map_data = map_no_pad;
for i = 1:length(map_data.reward_points_name)

change_day(i) = -50+ find(strcmp(reward_pos_all , map_data.reward_points_name{i}),1,"first");



end

x = 1:length(trial_score_all);
plot(x, movmean(trial_score_all,50));
hold on
xline(change_day)