clc
clear all
close all 

load("data/optimal_choice_percentage_matrix.mat")
load("data/reward_pos_trial.mat")
map_data = map_no_pad;

figure(1)
hold on
x = 1:size(choice_matrix ,2);
for i = 1:3 
   
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