clc
clear all
close all

load("data/Position_sequence_P.mat")
load("data/reward_pos_trial.mat")
map_data = map_no_pad;

%% 1 row: optimal , 2 row: sub optimal , 3: null 
choice_matrix = zeros(3, length(reward_pos_all));

for trial_num  = 1:length(reward_pos_all)
    single_trial = pos_sequence_all{trial_num};
    for time_num = 1:(-1+length(single_trial))
        next_node = single_trial{time_num + 1 };
        eval = node_evaluate(single_trial{time_num} , reward_pos_all{trial_num});

        if length(eval.index) == 1

            continue

        end

        [max_num, max_index] = max(eval.score);
        [min_num , min_index ] = min(eval.score);

        if strcmp(next_node , eval.index(max_index)) 
            choice_matrix(1, trial_num) = choice_matrix(1, trial_num) + 1;
        elseif strcmp(next_node , eval.index(min_index))
            choice_matrix(3, trial_num) = choice_matrix(3,trial_num) +1;

        else
            choice_matrix(2,trial_num) = choice_matrix(2,trial_num) + 1;
        end


    end

    choice_matrix(:,trial_num) = choice_matrix(:,trial_num)./(sum(choice_matrix(:,trial_num))); 

end
figure(1)
hold on
x = 1:length(pos_sequence_all);
for i = 1:3 
    choice_matrix(i,:) = movmean(choice_matrix(i,:) , 20);
    plot(x , choice_matrix(i,:) , 'LineWidth',2.2);
    

end

change_day = zeros(1, 9);

for i = 1:length(map_data.reward_points_name)

change_day(i) = -20+ find(strcmp(reward_pos_all , map_data.reward_points_name{i}),1,"first");

end
xline(change_day, 'LineWidth', 2)

xlim([0,519])
ylim([0,1])

xlabel('trial number')
ylabel('choice percentage')
legend("optimal" , "sub optimal" , "none sense")

save("data/optimal_choice_percentage_matrix" , "choice_matrix")




