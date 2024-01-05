function output = is_small_rewarded(pos_sequence)
map_data = map_no_pad;
output = struct('index' , [] , 'color' , []);

for i = 1:(length(pos_sequence)-1)
    [color, out] =  small_rewarded(pos_sequence{i} , pos_sequence{i+1}, map_data);
    if out 
        if isempty(output.index)
            output.index = i;
            output.color = color;

        else
            output.index = [output.index , i];
            output.color = [output.color , color];
        end

    end




end
end






function [color, out] = small_rewarded(char_1 , char_2, map_data)
pos_1 = map_data.important_points(alpha2num(char_1), :);
pos_2 = map_data.important_points(alpha2num(char_2), :);
out = 0;
color = 0;

for i = 1:length(map_data.sub_reward.color)
small_reward_pos = map_data.sub_reward.location(i,:);
if pos_1(1) == pos_2(1)
   if ~(pos_1(1) == small_reward_pos(1))
       continue
   end


   if min(pos_1(2), pos_2(2)) < small_reward_pos(2) && max(pos_1(2) , pos_2(2)) > small_reward_pos(2)
       out = 1;
       color = map_data.sub_reward.color{i};
       return
   end

elseif pos_1(2) == pos_2(2)
   if ~(pos_1(2) == small_reward_pos(2))
       continue
   end

   if min(pos_1(1) , pos_2(1)) < small_reward_pos(1) && max(pos_1(1) , pos_2(1)) > small_reward_pos(1)

       out = 1; 
       color = map_data.sub_reward.color{i};
       return
   end

end



end




end