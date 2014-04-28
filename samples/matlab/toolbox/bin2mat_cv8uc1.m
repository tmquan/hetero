function [img] = bin2mat_cv8uc1(filename, nRows, nCols, nTems)
%BIN2MAT Summary of this function goes here
%   Detailed explanation goes here
    fid = fopen(filename, 'rb');
    same_real = fread(fid, nRows*nCols*nTems, '*uint8');

    img = same_real;
    fclose(fid);
    img = reshape(img, [nRows, nCols, nTems]);

end

