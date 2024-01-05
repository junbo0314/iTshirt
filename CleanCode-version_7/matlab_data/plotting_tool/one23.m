clc
clear all
close all

load("data/P_OFC.mat")
pad = 1000 
pos_all = [];
for trial_num = 1:519 
    
    pos = [bhv{trial_num}(pad,1) , bhv{trial_num}(pad,2)];
    pos_all = [pos_all ; pos];
end

show_map
hold on
scatter(pos_all(:,1) , pos_all(:,2))