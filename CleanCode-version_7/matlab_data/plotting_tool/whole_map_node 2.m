clc
clear all 
close all 

map_data = map;
load("data/Position_sequence_P.mat")

for i = 1:length(pos_sequence_all)
    seq = pos_sequence_all{i};

    for n = 1:length(seq)-1
        if strcmp(seq(n) , 'B');
            [seq(n) , seq(n+1)];
            if ~strcmp(seq(n+1) , 'F');
                i
            end


        end
    end
end