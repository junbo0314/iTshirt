clc
clear all
close all

map_data = map;
parfor i = 1:8
trial_gifs(map_data.important_points_name{i});

end