function output = Optimal_path_sorting(start_position , reward_position)
output = struct();



map_data = map;

reward_index = find(strcmp(map_data.reward_points_name , reward_position));

Optimal_reward_pos = [];
Second_Optimal_pos = [];


if reward_index == 1
    Optimal_reward_pos = 1;
    Second_Optimal_pos = 1;
elseif reward_index == 2
    Optimal_reward_pos = 2;
    Second_Optimal_pos = 3;
elseif reward_index == 3
    Optimal_reward_pos = 4;
    Second_Optimal_pos = 4;
elseif reward_index == 4
    Optimal_reward_pos = 5;
    Second_Optimal_pos = 6;
elseif reward_index == 5
    Optimal_reward_pos = 6;
    Second_Optimal_pos = 10;
elseif reward_index == 6
    Optimal_reward_pos = 3;
    Second_Optimal_pos = 8;
elseif reward_index == 7
    Optimal_reward_pos = 10;
    Second_Optimal_pos = 9;
elseif reward_index == 8 || reward_index ==9
    Optimal_reward_pos = 11;
    Second_Optimal_pos = 10;
end

map_data = map;

Optimal_path_all = optimal_path(start_position , map_data.important_points_name{Optimal_reward_pos}); 
path_num = 1;
sub_rewarded_path_num = 1;

for i = 1:size(struct2table(Optimal_path_all),2)
    path = Optimal_path_all.(append('path_' , int2str(i)));
    if sub_rewarded(path)
        output.optimal_path.sub_rewarded.(append('path_' , int2str(sub_rewarded_path_num))) = path;
        sub_rewarded_path_num = sub_rewarded_path_num + 1;
    else
        output.optimal_path.no_sub.(append('path_' , int2str(path_num) )) = path;
        path_num = path_num + 1; 
    end
end


if ~(Optimal_reward_pos == Second_Optimal_pos)

    Optimal_path_all = optimal_path(start_position , map_data.important_points_name{Second_Optimal_pos}); 
    path_num = 1;
    sub_rewarded_path_num = 1;

    for i = 1:size(struct2table(Optimal_path_all),2)
       path = Optimal_path_all.(append('path_' , int2str(i)));
       if sub_rewarded(path)
         output.sub_optimal_path.sub_rewarded.(append('path_' , int2str(sub_rewarded_path_num))) = path;
         sub_rewarded_path_num = sub_rewarded_path_num + 1;
       else
         output.sub_optimal_path.no_sub.(append('path_' , int2str(path_num) )) = path;
         path_num = path_num + 1; 
       end
    end


    try output.sub_optimal_path;
        try 
            path_length = size(output.sub_optimal_path.no_sub.path_1,2);
        catch 
            path_length = size(output.sub_optimal_path.sub_rewarded.path_1 ,2);
        end

        try 
            orginal_length = size(output.optimal_path.no_sub.path_1 , 2);
        catch
            orginal_length = size(output.optimal_path.sub_rewarded.path_1 , 2);
        end

        if path_length < orginal_length
            mem = output.optimal_path;
            output.optimal_path = output.sub_optimal_path;
            output.sub_optimal_path = mem;
        end



    catch


    end






end
end

function out = sub_rewarded(cell)
out = false;
small.reward_1 = {'B' , 'E'};
small.reward_2 = {'F' , 'J'};

for reward_num = 1:2
    if (sum(strcmp(cell , small.(append('reward_' , int2str(reward_num))){1})))...
            && (sum(strcmp(cell , small.(append('reward_' , int2str(reward_num))){2})))
        
        pos1 = find(strcmp(cell , small.(append('reward_' , int2str(reward_num))){1}));
        pos2 = find(strcmp(cell , small.(append('reward_' , int2str(reward_num))){2}));
        if abs(pos1 - pos2) == 1
            out = true;
            return
        end

    end

    
end
end


