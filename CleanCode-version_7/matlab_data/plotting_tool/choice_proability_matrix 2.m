clc
clear all
close all
load("data/Position_sequence_P.mat")
map_data = map();
places = map_data.important_points_name;


window_size = 50;

proability_matrix_all = cell(length(1: (length(pos_sequence_all) - window_size+1)), 1);

for window_index = 1: (length(pos_sequence_all) - window_size+1)
    proability_matrix = zeros(length(places));
    windowed_data = pos_sequence_all(window_index:(window_index + window_size - 1));

    for single_window_index = 1:length(windowed_data)
        single_windowed_data = windowed_data{single_window_index};
        
    


        for time = 1:(length(single_windowed_data)-1)
         

            proability_matrix(alpha2num(single_windowed_data{time}) , alpha2num(single_windowed_data{time+1})) ...
                = proability_matrix(alpha2num(single_windowed_data{time}) , alpha2num(single_windowed_data{time+1})) +1 ;


        end


    end


    for i = 1:length(places)
        weight = sum(proability_matrix(i,:));
        if weight == 0
            continue
        end

        proability_matrix(i,:) = proability_matrix(i,:)/weight;


    end

    proability_matrix_all{window_index} = proability_matrix;

end


save("data/proability_choice_matrix.mat" , "proability_matrix_all");



