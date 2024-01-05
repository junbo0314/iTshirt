function output = single_node_map(node , draw)
close all


load("data/proability_choice_matrix.mat")
load("data/Choice_matrix.mat")
load("data/reward_pos_trial.mat")
map_data = map;

change_day = zeros(1, 9);

for i = 1:length(map_data.reward_points_name)

change_day(i) = find(strcmp(reward_pos_all , map_data.reward_points_name{i}),1,"first");


end


map_data = map();
node_index = map_data.important_points_name;



node_data = cell(length(proability_matrix_all) , 1);

data_exist = zeros(1,length(proability_matrix_all{1}));
data_exist_index = [];



for i = 1:length(node_data)
    node_data{i} = proability_matrix_all{i}(alpha2num(node) , :);
    data_exist = data_exist + node_data{i};
    
end

data_exist_index = find(data_exist);
legend_index = cell(1,length(data_exist_index));

for legend_i = 1:length(data_exist_index)
    legend_index{legend_i} = up_down_left_right(alpha2num(node) , data_exist_index(legend_i) , map_data);


end

node_data = cell2mat(node_data).';

if draw

figure(1)
hold on

for i = 1:length(data_exist_index)
plot(1:470 , movmean(node_data(data_exist_index(i) , :) , 10),'LineWidth',  2.2)
end
legend(legend_index, 'AutoUpdate','off')
xline(change_day,"LineWidth",1.8)

xlabel("Trial num")
ylabel("Persentage %")
ylim([-0.1,1.5])

end

max_data = zeros(1,length(proability_matrix_all));

for i = 1:length(proability_matrix_all)
    data_window = node_data(data_exist_index ,:);
    [max_num , max_index] = max(data_window(:,i));
    without_max = data_window(:,i);
    without_max(max_index) = [];
    without_max(without_max == 0) = [];
    if isempty(without_max)
        without_max = 0;
    end
   

    max_data(i) = max_num - min(without_max);

end

if ~draw 

figure(2)
plot(1:470 , movmean(max_data,10) , 'LineWidth',1.6)
hold on
xline(change_day);


xlabel("Trial num")
ylabel("Persentage %")
ylim([-0.1,1.5])

end

output = max_data;

end