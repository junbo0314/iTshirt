clc
clear all 
close all

load("data/Position_sequence_P.mat")
trial_num = 213;

data = pos_sequence_all{trial_num};
new_data = [data(1:56) , 'E' , data{end} ]

pos_sequence_all{trial_num} = new_data;

save("data/Position_sequence_P.mat" , 'pos_sequence_all')