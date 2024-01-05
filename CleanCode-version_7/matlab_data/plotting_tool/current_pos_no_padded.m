function output = current_pos_no_pad(position)
map_data = map_no_pad;
pos_index = map_data.important_points_name;

for n = 1:(length(pos_index)+1)
    if n == (length(pos_index)+1)
        output = "Z";
        return
    end
    if position(1) >= ((map_data.important_points(n,1) - 1.5) )&& (position(1) <= (map_data.important_points(n,1) + 1.5))
      if (position(2) >= (map_data.important_points(n,2) - 1.5)) && (position(2) <= (map_data.important_points(n,2) + 1.5))
        output = pos_index(n);
        return
      end
    end




end





end