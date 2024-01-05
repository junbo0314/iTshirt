function output = alpha2num(alphabat)
 alpha_vec = {'A' , 'B' , 'C' ,'D' , 'E' , 'F', 'G' , 'H' ,'I' ,'J' , 'K' , 'L' ,'M', 'N','O' , 'P' , 'Q' , 'R' , 'S' , 'T' , 'U' , 'V' ,'W'};
 output = find(strcmp(alpha_vec, alphabat));

 if isempty(output)
    alpha_vec = {'a' , 'b' , 'c' ,'d' ,'e' ,'f' ,'g' , 'h', 'i'};
    output = find(strcmp(alpha_vec, alphabat));

 end



end