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

idx = 800

for trial_num = 1:519 
    pos = [bhv{trial_num}(idx,1), bhv{trial_num}(idx,2)];
    scatter(pos(1) , pos(2) , 50, "red" , 'o')


end