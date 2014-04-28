function [] = imwritetif(mat, filename)
    dim = size(mat)
	for k=1:dim(3)
		if (k==1)
			imwrite(uint8(mat(:, :, k*3)), filename, 'tif', 'Compression','none');
		else
			imwrite(uint8(mat(:, :, k*3)), filename, 'tif', 'WriteMode', 'append', 'Compression','none');
		end
	end
end