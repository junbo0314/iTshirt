function output = node_evaluate(node, reward)



output = struct('index' , [] , 'score' , []);
map_data = map_no_pad;
end_point = possible_end(reward);





near = near_points(node , map_data);



for i = 1:length(near)
    score = 0; 
    single_near = near{i};
    for n = 1:2
        path_all = optimal_path(single_near , end_point{n});
        for k = 1:size(struct2table(path_all) ,2)
            single_path = path_all.(append("path_" , int2str(k)));
            if is_in_path(single_path , end_point(3-n))
                continue
            end
            single_path;
            new_score = path_evaluate(single_path);
            if new_score > score
                score = new_score;

            end





        end


    end

    if i == 1 
        output.index = char(near{1});
    else

       output.index(i) = char(near{i});
    end
    output.score(i) = score;



end










end


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


function output = is_in_path(path , node)
output = 0;
if sum(strcmp(path, node))
    output = 1;
end




end


