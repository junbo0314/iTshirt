function show_map()
figure()
hold on
map_data = map_no_pad;




scatter(map_data.map(:,1) , map_data.map(:,2))

scatter(map_data.important_points(:,1) , map_data.important_points(:,2),100, 'red' , 'filled' , 'o' , 'LineWidth', 1.5)

scatter(map_data.reward_points(:,1) , map_data.reward_points(:,2) ,100, 'yellow' , 'filled' , 'O' , 'LineWidth',1.5)

for i = 1:length(map_data.sub_reward.name)
    scatter(map_data.sub_reward.location(i,1) , map_data.sub_reward.location(i,2) ,100, map_data.sub_reward.color{i} , 'filled' , 'o')

end

end