function y = scale255(x)
%% Scale to the 8bit display images
% Range of pixel 0..255
x = abs(x);
maxVal = max(x(:));
minVal = min(x(:));

y = (x - minVal)./(maxVal-minVal)*255;
end