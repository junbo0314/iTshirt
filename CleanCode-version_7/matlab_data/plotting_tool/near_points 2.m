
function output = near_points(node, map_data)
output = {};
node_cord = map_data.important_points(alpha2num(node),:);
udlf = [0,14;0,-14;-14,0;14,0];
for i = 1:size(udlf , 1)
    index = 0;
    new_cord = node_cord + udlf(i,:);
    
    index = find(sum((new_cord == map_data.important_points).') == 2);

    if index
        if on_road(node_cord +(1/2)*udlf(i,:), map_data)
        if isempty(output)
            output{1} = char(map_data.important_points_name{index});
        else
            output{end+1} = char(map_data.important_points_name{index});
        end
        end
    end



end


end