function mat2bin_cv32fc1(mat, filename)
    [nRows nCols nTems] = size(mat);
    
    % Create the file
    fid = fopen(filename, 'wb');
        fwrite(fid, mat, 'single');
    fclose(fid);

end

