clc
clear all
close all
addpath("data")
load("P_OFC.mat")
close all
trial_num = 200;

filename = 'test.gif';

map_data = map;

fig = figure(1);
title('VR maze')
scatter(map_data.map(:,1) , map_data.map(:,2));
hold on 

for idx = 1000:200:size(bhv{trial_num} , 1)

    
   
    pos = [bhv{trial_num}(idx,1), bhv{trial_num}(idx,2)];
    pos_index = current_pos(pos);
    scatter(pos(1) , pos(2) , 50, "red" , 'o')
    try delete(hText)
    catch
    end
    if ~strcmp(pos_index,"Z")
        hText = text(30, 30 , append('pos is in ', pos_index));


    end
    drawnow
    frame = getframe(fig);
    im{idx} = frame2im(frame);



   

end


