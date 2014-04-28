function [img] = bin2mat_cv32fc1(filename, nRows, nCols, nTems)
%BIN2MAT Summary of this function goes here
%   Detailed explanation goes here
    fid = fopen(filename, 'rb');
    same_real = fread(fid, [nRows*nCols*nTems], '*single',0,'ieee-be'); %big endian

    img = same_real;
    fclose(fid);
    img = reshape(img, [nRows, nCols, nTems]);
	img = img';
end
