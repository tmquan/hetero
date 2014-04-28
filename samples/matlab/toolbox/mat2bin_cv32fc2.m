function mat2bin_cv32fc2(data, filename)
    dim   = size(data);
    %%

    interleaved = zeros(dim);

    %% Alternate real and imaginary data
    newx = 1;
    for x = 1:dim(1)
        interleaved(newx + 0,:,:,:) = real(data(x,:,:,:));
        interleaved(newx + 1,:,:,:) = imag(data(x,:,:,:));
        newx = newx + 2;
    end

    %% Create the file
    fid = fopen(filename, 'wb');
        fwrite(fid, interleaved, 'single');
    fclose(fid);
end

