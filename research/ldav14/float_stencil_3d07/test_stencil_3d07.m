clc;
clear all;
close all;
addpath(genpath('.'));
%%
% cmd = 'mpirun -np 1 --host node002 ../../../build/bin/test_float_stencil_3d07';
diary
for bz = [4, 8, 16, 32, 64]
    for by = [4, 8, 16, 32, 64]
        for bx = [4, 8, 16, 32, 64]
            cmd = sprintf('nvprof --metrics flops_sp ../../../build/bin/test_float_stencil_3d07 -bx=%d -by=%d -bz=%d -num=1', bx, by, bz);
			system(cmd);	
        end
    end
end