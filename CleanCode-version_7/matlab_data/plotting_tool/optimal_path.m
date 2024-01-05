function output = optimal_path(starting_point , ending_point)


optimal_path_data = struct();
path_num = 1;

Corner = struct();
map_data = map_no_pad;
starting_point_mem = [];

null_node = {'A' , 'B' , 'C' , 'G' , 'H',  'Q' , 'R' ,'V' ,'W'};
if sum(strcmp(null_node , starting_point))
    starting_point_mem = char(starting_point);
    starting_point = char(near_points(starting_point, map_data));

end


for i = 1:length(map_data.important_points_name)
Corner.(map_data.important_points_name{i}) = map_data.important_points(i,:);

end

delta_vec = Corner.(char(ending_point)) - Corner.(char(starting_point));


x_choice_num = abs(delta_vec(1)/14);
y_choice_num = abs(delta_vec(2)/14);


if ((x_choice_num + y_choice_num) == 1)
    optimal_path_data.path_1 = {starting_point , ending_point};
    output = optimal_path_data;
    return
elseif ((x_choice_num + y_choice_num) == 0)
    optimal_path_data.path_1 = {starting_point};
    output = optimal_path_data;
    return
end


for i = 1:((factorial(x_choice_num + y_choice_num))/(factorial(x_choice_num)*factorial(y_choice_num)))
path_xy = cell(1,x_choice_num+y_choice_num);
path_xy(:) = {'y'};
path = {starting_point};
Choice_position = 1:length(path_xy);
Choice = nchoosek(Choice_position , x_choice_num);
Choice = Choice(i,:);

for n = 1:length(Choice)
    path_xy(Choice(n)) = {'x'};
end

curr_pos = Corner.(char(starting_point));

for k = 1:length(path_xy)
next_pos_mov = zeros(1,2);
if strcmp(path_xy(k), 'x')
   next_pos_mov(1) = (delta_vec(1)/(abs(delta_vec(1))))*14;
else
    next_pos_mov(2) = (delta_vec(2)/(abs(delta_vec(2))))*14;
end

curr_pos = curr_pos + next_pos_mov;
curr_pos_index = [];
for position_num = 1:length(map_data.important_points_name)
    if curr_pos == map_data.important_points(position_num,:)
        curr_pos_index = map_data.important_points_name(position_num);
    end
end

if isempty(curr_pos_index) || ~on_road( curr_pos - next_pos_mov/2, map_data)
    ER = 1;
    break 
end



path(end + 1 ) = curr_pos_index;

end

try ER;
    clearvars ER;
    continue 
catch
end

optimal_path_data.(append('path_' , int2str(path_num))) = path;
path_num = path_num + 1; 

end

if ~isempty(starting_point_mem)
    for i = 1:(path_num-1)
    optimal_path_data.(append('path_' , int2str(i))) = [starting_point_mem , optimal_path_data.(append('path_' , int2str(i)))];
    end


end


output = optimal_path_data;
end


