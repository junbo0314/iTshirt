fun
ction trial_gifs(node)
addpath('data')
load("data/proability_choice_matrix.mat")
load("data/reward_pos_trial.mat")
file_name = append("gifs/trial_changes_" , char(node) , ".gif");
map_data = map;


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
figure('Position', [1,1,960,540]);
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

scatter(map_data.important_points(node_num,1) , map_data.important_points(node_num,2),100, 'red' , 'filled' , 'o' , 'LineWidth', 5)

scatter(map_data.reward_points(reward_num,1) , map_data.reward_points(reward_num,2) , 100,'yellow' , 'filled' , 'O' , 'LineWidth',5)

for i = 1:length(map_data.sub_reward.name)
    scatter(map_data.sub_reward.location(i,1) , map_data.sub_reward.location(i,2) , 50, map_data.sub_reward.color{i} , 'filled' , 'o')

end

title("Map")

nexttile(fig , 2)







node_data_trial = node_data;




hold on

for i = 1:length(data_exist_index)

data =  movmean(node_data_trial(data_exist_index(i) , :) ,10 );

plot(1:trial_num ,data(:,1:trial_num), 'LineWidth',...
    3.5 ,'Color', Color.(char(legend_index{i})))
end
lge = legend(legend_index, 'AutoUpdate','off');
lge.FontSize = 20;
xline(change_day,"LineWidth",2.2)

xlabel("Trial num")
ylabel("Persentage %")
xlim([0, 500])
ylim([-0.1,1.5])
set(gca , 'YTick' , -0.1:0.1:1);

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
end