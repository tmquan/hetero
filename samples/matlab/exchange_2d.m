clc;
clear all;
close all;
%%
addpath(genpath('.'));
addpath(genpath('/home/tmquan/hetero/build/samples/hybrid/exchange_2d/'));

dimx = 500;
dimy = 450;

px   = 4;
py   = 4;
%%
% rawfilename = sprintf('result_%02d_%02d.raw', x-1, y-1);
rawfilename = sprintf('/home/tmquan/hetero/data/barbara_%03dx%03d.raw', dimx, dimy);
image = bin2mat_cv32fc1(rawfilename, dimx, dimy, 1);

image = uint8(image);
imgfilename = sprintf('barbara_%02dx%02d.png', dimx, dimy);
imwrite(image, imgfilename);
%%
for y=1:py
	for x=1:px
		rawfilename = sprintf('result_%02d_%02d.raw', x-1, y-1);
		if y==py 
			ty = mod(dimy, 128)
		else
			ty = 128
		end
		
		if x==px
			tx = mod(dimx, 128)
		else
			tx = 128
		end
		
		% ty = 128;
		% tx = 128;
		
		image = bin2mat_cv32fc1(rawfilename, tx, ty, 1);
		image = uint8(image);
		imgfilename = sprintf('result_%02d_%02d.png', x-1, y-1);
		imwrite(image, imgfilename);
	end
end

%%
halo = 12;
for y=1:4
	for x=1:4
		
		if y==py 
			ty = mod(dimy, 128)
		else
			ty = 128
		end
		
		if x==px
			tx = mod(dimx, 128)
		else
			tx = 128
		end
		
		% ty = 128;
		% tx = 128;
		
		if (y==1) || (y== py)
			ty = ty+1*halo;
		else
			ty = ty+2*halo;
		end
		
		if (x==1) || (x== px)
			tx = tx+1*halo;
		else
			tx = tx+2*halo;
		end
		% zero
		rawfilename = sprintf('extend_%02d_%02d.raw', x-1, y-1);
		image = bin2mat_cv32fc1(rawfilename, tx, ty, 1);
		image = uint8(image);
		imgfilename = sprintf('extend_%02d_%02d.png', x-1, y-1);
		imwrite(image, imgfilename);
		
		rawfilename = sprintf('extend_left2right_%02d_%02d.raw', x-1, y-1);
		image = bin2mat_cv32fc1(rawfilename, tx, ty, 1);
		image = uint8(image);
		imgfilename = sprintf('extend_left2right_%02d_%02d.png', x-1, y-1);
		imwrite(image, imgfilename);
		
		rawfilename = sprintf('extend_right2left_%02d_%02d.raw', x-1, y-1);
		image = bin2mat_cv32fc1(rawfilename, tx, ty, 1);
		image = uint8(image);
		imgfilename = sprintf('extend_right2left_%02d_%02d.png', x-1, y-1);
		imwrite(image, imgfilename);
		
		rawfilename = sprintf('extend_top2bottom_%02d_%02d.raw', x-1, y-1);
		image = bin2mat_cv32fc1(rawfilename, tx, ty, 1);
		image = uint8(image);
		imgfilename = sprintf('extend_top2bottom_%02d_%02d.png', x-1, y-1);
		imwrite(image, imgfilename);
		
		rawfilename = sprintf('extend_bottom2top_%02d_%02d.raw', x-1, y-1);
		image = bin2mat_cv32fc1(rawfilename, tx, ty, 1);
		image = uint8(image);
		imgfilename = sprintf('extend_bottom2top_%02d_%02d.png', x-1, y-1);
		imwrite(image, imgfilename);
	end
end
