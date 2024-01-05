function output = map_no_pad()



output = struct();
line = [];
starting_point = [-20 , 21; -34 , 7; -22, -7; -34, -21; -21,-34; -7 , -34;7, -22; 21 , -34];
ending_point = [8,21; 22,7; 33, -7; 22,-21;-21,7; -7, 22; 7,33; 21 ,19];
padded = 0;
output.important_points = [-35+ padded, -21; -35+padded, 7; -21 , -35+padded; -21 , -21; -21, -7; -21, 7; -21+padded , 21; -7,-35+padded;  -7 , -21; -7 , -7; -7 , 7; -7 , 21;...
                                    7 , -21; 7, -7; 7 , 7; 7 ,21 ; 7,35-padded; 21,-35+padded; 21 , -21; 21 , -7 ; 21 , 7; 21 ,21-padded; 35-padded , -7];
output.important_points_name = {'A' , 'B' , 'C' ,'D' , 'E' , 'F', 'G' , 'H' ,'I' ,'J' , 'K' , 'L' ,'M', 'N','O' , 'P' , 'Q' , 'R' , 'S' , 'T' , 'U' , 'V' ,'W'};
output.reward_points_name = {'a' , 'b' , 'c', 'd', 'e', 'f', 'g', 'h' , 'i'};
reward_pos_all = load("./data/reward_pos.mat");
output.reward_points = reward_pos_all.reward_pos_all;

output.sub_reward.name = {'a' , 'b' ,'c' ,'d' , 'e', 'f', 'g' , 'h'};
output.sub_reward.color = {'g' , 'g' , 'g' ,'g' ,'k', 'k', 'k', 'm'};
output.sub_reward.location = [-18,-7; -18 , 21; 4,7;  7,30; -30 , -21; -30,7; 18,-7;10,-21];


for n = 1:size(starting_point , 1)
    line = [line ; draw_line(starting_point(n,:) , ending_point(n,:))];
end
output.map = line;
end




function output = draw_line(start, finish)


if start(1) == finish(1)
line = start(2):0.01:finish(2);
line = [start(1)*ones(1, length(line)) ; line].';
elseif start(2) == finish(2)
line = start(1):0.01:finish(1);
line = [line; start(2)*ones(1,length(line))].';
end

output = line;



end