#!/bin/bash
yes | pkilluser `whoami` 

# Release nvprof 
pexec "rm -rf /tmp/.nvprof/nvprof.lock"

# Print GPU status from the head and execute from nodes
pexec "nvidia-smi"

for bz in 4, 8, 16, 32, 64
do
	for by in 4, 8, 16, 32, 64
	do
		for bx in 4, 8, 16, 32, 64
		do
			# echo $'--------------------------------------------'
			cmd=$'pexec -n=node001 -t=0
			"nvprof --print-gpu-trace 
			 ../../../build/bin/test_float_stencil_3d07 
			-dx=512 -dy=512 -dz=512 -bx=$bx -by=$by -bz=$bz -num=1\n"'
			echo $cmd
			eval $cmd
			echo $'\n'
			
			cmd=$'pexec -n=node002 -t=0 
			"nvprof --metrics flops_sp,gld_throughput,gst_throughput,sm_efficiency,achieved_occupancy 
			 ../../../build/bin/test_float_stencil_3d07 
			-dx=512 -dy=512 -dz=512  -bx=$bx -by=$by -bz=$bz -num=1\n"'
			echo $cmd
			eval $cmd
			echo $'\n'
		done
	done
done
