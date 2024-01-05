function output = on_road(coordinate , map_data)
output = logical(sum(sum((coordinate == map_data.map).') == 2));
end
