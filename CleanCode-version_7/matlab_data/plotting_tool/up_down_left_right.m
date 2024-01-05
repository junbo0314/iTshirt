function output = up_down_left_right(before, after, map_data)
    
    if ~isnumeric(before)
    

    before = find(strcmp(before , map_data.important_points_name));
    after  = find(strcmp(after , map_data.important_points_name));

    end

    delta = map_data.important_points(after , :) - map_data.important_points(before,:);
    if delta(1) > 0 
        output = 'Right';
    elseif delta(1) < 0 
        output = 'Left';
    elseif delta(2) > 0
        output = 'Up';
    elseif delta(2) < 0
        output = 'Down';
    else
        error("error!");
    end








end