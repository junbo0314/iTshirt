clc
clear all
close all

load("data/proability_choice_matrix.mat")
load("data/reward_pos_trial.mat")
file_name = "trial_chang.gif";
map_data = map;
node = 'D';

node_data = cell(length(proability_matrix_all) , 1);
data_exist = zeros(1,length(proability_matrix_all{1}));
data_exist_index = [];

Color = struct('Up' , [57, 106 , 177]./255 , 'Down' , [204, 37 , 41]./255 , ...
              'Left' , [62, 150, 81]./255 , 'Right' , [107, 76, 154]./255);


for i = 1:length(node_data)
    node_data{i} = proability_matrix_all{i}(alpha2num(node) , :);
    data_exist = data_exist + node_data{i};
    
end

node_data = cell2mat(node_data).';

data_exist_index = find(data_exist);
legend_index = cell(1,length(data_exist_index));

for legend_i = 1:length(data_exist_index)
    legend_index{legend_i} = up_down_left_right(alpha2num(node) , data_exist_index(legend_i) , map_data);


end
figure('Position', get(0,"ScreenSize"));
fig = tiledlayout(1,2,'Padding' , 'none');


for trial_num = 1:length(node_data)




reward_num = alpha2num(char(reward_pos_all{trial_num+50}));
node_num = alpha2num(node);


change_day = zeros(1, 9);

for i = 1:length(map_data.reward_points_name)

change_day(i) = -50 + find(strcmp(reward_pos_all , map_data.reward_points_name{i}),1,"first");



end






nexttile(fig,1)
hold on

scatter(map_data.map(:,1) , map_data.map(:,2), 'b')

scatter(map_data.important_points(node_num,1) , map_data.important_points(node_num,2), 'red' , 'filled' , 'o' , 'LineWidth', 2.5)

scatter(map_data.reward_points(reward_num,1) , map_data.reward_points(reward_num,2) , 'yellow' , 'filled' , 'O' , 'LineWidth',2.5)

title("Map")

nexttile(fig , 2)







node_data_trial = node_data;




hold on

for i = 1:length(data_exist_index)

data =  movmean(node_data_trial(data_exist_index(i) , :) ,10 );

plot(1:trial_num ,data(:,1:trial_num), 'LineWidth',...
    2.2 ,'Color', Color.(char(legend_index{i})))
end
legend(legend_index, 'AutoUpdate','off')
xline(change_day,"LineWidth",1.8)

xlabel("Trial num")
ylabel("Persentage %")
xlim([0, 500])
ylim([-0.1,1.5])

hold off

drawnow

frame = getframe(1);
img = frame2im(frame);
[imind cm] = rgb2ind(img,256);

if trial_num ==1
    imwrite(imind , cm, file_name , 'gif' ,'LoopCount',  1 , 'DelayTime', 1/24);
else
    imwrite(imind,cm,file_name,'gif','WriteMode','append','DelayTime',1/24);
end

end