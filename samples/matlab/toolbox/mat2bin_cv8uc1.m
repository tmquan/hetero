function mat2bin_cv8uc1(mat, filename)
    [nRows nCols nTems] = size(mat);
    
    tmp = zeros(nRows, nCols, nTems);
    for k=1:nTems
        tmp(:,:,k) = mat(:,:,k); % Transpose the image is very important
    end
    % Create the file
    fid = fopen(filename, 'wb');
        %fwrite(fid, mat, 'uint8');
        fwrite(fid, tmp, 'uint8');
        
    fclose(fid);

end

