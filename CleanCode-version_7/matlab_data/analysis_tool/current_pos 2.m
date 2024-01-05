function output = current_pos(position)
map_data = map;
pos_index = map_data.important_points_name;
pad = 2;
for n = 1:(length(pos_index)+1)
    if n == (length(pos_index)+1)
        output = "Z";
        return
    end
    if position(1) >= ((map_data.important_points(n,1) - pad) )&& (position(1) <= (map_data.important_points(n,1) + pad))
      if (position(2) >= (map_data.important_points(n,2) - pad)) && (position(2) <= (map_data.important_points(n,2) + pad))
        output = pos_index(n);
        return
      end
    end




end





end