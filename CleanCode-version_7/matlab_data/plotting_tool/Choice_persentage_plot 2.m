clc
clearvars
close all

load("data/Choice_matrix.mat")
load("data/reward_pos_trial.mat")
map_data = map;
x = 1:size(Choice_matrix_all , 1);
change_day = zeros(1, 9);

for i = 1:length(map_data.reward_points_name)

change_day(i) = find(strcmp(reward_pos_all , map_data.reward_points_name{i}),1,"first");


end

figure(1)
hold on
plot(x, movmean(Choice_matrix_all(:,1) - Choice_matrix_all(:,4) , 10) , "LineWidth", 2)
plot(x, movmean(Choice_matrix_all(:,1) - Choice_matrix_all(:,2) + Choice_matrix_all(:,4), 10) , "LineWidth",1.6);


xline(change_day)

xlim([0 , length(x)])
ylim([0,100])


figure(2) 
hold on

plot(x, movmean(Choice_matrix_all(:,1) + Choice_matrix_all(:,3) , 10) , "LineWidth", 1.5)
plot(x,movmean(Choice_matrix_all(:,2) + Choice_matrix_all(:,4) , 10) , "LineWidth", 1.5 )
plot(x, movmean(Choice_matrix_all(:,5) , 10) , "LineWidth",1.5)


legend('1' , '2' ,'3')
xline(change_day)

xlim([0 , length(x)])
ylim([0,100])


figure(3)

plot(x , -movmean(Choice_matrix_all(:,1) + Choice_matrix_all(:,3) , 10) +movmean(Choice_matrix_all(:,2) + Choice_matrix_all(:,4) , 10) , "LineWidth",1.7 )


xline(change_day)

xlim([0 , length(x)])
ylim([0,100])
