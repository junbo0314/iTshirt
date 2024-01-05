function reward_pos_close_all = reward_position(reward)
map_data = map_no_pad;


reward_cordinate_all = map_data.reward_points;
important_point = map_data.important_points;

target_reward_index = find(strcmp(reward , map_data.reward_points_name));

target_reward_cor = reward_cordinate_all(target_reward_index , :);

if mod(target_reward_cor(1) , 7) == 0
    pos_reward_pos = zeros(2,2);
    pos_reward_pos(1,:) = [target_reward_cor(1) , 14*ceil((target_reward_cor(2)-7)/14)+7 ];
    pos_reward_pos(2,:) = [target_reward_cor(1) , 14*floor((target_reward_cor(2)-7)/14)+7];

else
    pos_reward_pos = zeros(2,2);
    pos_reward_pos(1,:) = [14*ceil((target_reward_cor(1)-7)/14) + 7 , target_reward_cor(2) ];
    pos_reward_pos(2,:) = [14*floor((target_reward_cor(1)-7)/14) + 7,target_reward_cor(2)];


end
reward_pos_close_all = {};
for i = 1:2 
    if strcmp(current_pos_no_padded(pos_reward_pos(i,:)) , 'Z')
        continue
    end

   reward_pos_close = current_pos_no_padded(pos_reward_pos(i,:));

   

   if isempty(reward_pos_close_all)
       reward_pos_close_all = reward_pos_close;


   else
       reward_pos_close_all{end+1} = char(reward_pos_close);
   end
end


end
