function output = possible_end(reward)

output = {};
map_data = map_no_pad; 
pos_of_reward = map_data.reward_points(alpha2num(reward),:);

for i = 1:length(map_data.important_points_name)
    node = map_data.important_points(i,:);

    if node(1) == pos_of_reward(1) && (abs(pos_of_reward(2)-node(2)) < 14)
        if isempty(output)
            output = {char(map_data.important_points_name{i})};
        else
            output{end+1} = char(map_data.important_points_name{i});
        end
    elseif node(2) == pos_of_reward(2) && (abs(pos_of_reward(1) - node(1)) < 14)
        if isempty(output)
            output = {char(map_data.important_points_name{i})};
        else
            output{end+1} = char(map_data.important_points_name{i});
        end
    end







end