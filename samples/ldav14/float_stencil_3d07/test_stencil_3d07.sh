#!/bin/bash
# yes | pkilluser `whoami` 

# Release nvprof 
# pexec "rm -rf /tmp/.nvprof/nvprof.lock"

# Print GPU status from the head and execute from nodes
# pexec "nvidia-smi"
for (( ilp=16; ilp>=1; ilp/=2 ))
do
	for (( bz=1; bz<=1; bz+=1 ))
	do
		for (( by=2; by<=16; by+=2 ))
		do
			for (( bx=16; bx<=64; bx*=2 ))
			do
				printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
				printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
				cmd=$'
				 ../../../build/bin/test_float_stencil_3d07 
				-dx=512 -dy=512 -dz=512 -bx=$bx -by=$by -bz=$bz -ilp=$ilp -num=1\n'
				echo $cmd
				eval $cmd
				echo $'\n'
			done
		done
	done
done
