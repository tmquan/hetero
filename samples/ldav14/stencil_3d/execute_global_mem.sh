#!/bin/bash

touch test_stencil_3d_global.log
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' - > test_stencil_3d_global.log
# for (( ilp=1; ilp<=512; ilp*=2 ))
for (( ilp=8; ilp<=16; ilp*=2 ))
do
	for (( bz=1; bz<=1; bz+=1 ))
	do
		for (( bx=1; bx<=512; bx*=2 ))
		do
			for (( by=512/bx; by<=512/bx; by*=2 ))
			do
				str=$"
				 bx=$bx, by=$by, bz=$bz, ilp=$ilp"
				echo $str
				# Create a new source code
				
				# touch global_mem.cu
				# rm global_mem.cu
				# rm test_stencil_3d_global
				
				# Construct the code
				echo "" > global_mem.cu
				echo "#define DIMX 512" >> global_mem.cu
				echo "#define DIMY 512" >> global_mem.cu
				echo "#define DIMZ 512" >> global_mem.cu

				echo "#define ILP  $ilp" >> global_mem.cu


				echo "#define BLKX $bx" >> global_mem.cu
				echo "#define BLKY $by" >> global_mem.cu
				echo "#define BLKZ $bz" >> global_mem.cu
				
				
				# Concatenate the remain
				cat template_global_mem.cu >> global_mem.cu

				# Compile
				nvcc -arch=sm_20 global_mem.cu -o test_stencil_3d_global

				#Redirect and display
				./test_stencil_3d_global >> test_stencil_3d_global.log
			done
		done
	done
done

echo "" > global_mem.cu
echo "#define DIMX 512" >> global_mem.cu
echo "#define DIMY 512" >> global_mem.cu
echo "#define DIMZ 512" >> global_mem.cu

echo "#define ILP  8" >> global_mem.cu


echo "#define BLKX 8" >> global_mem.cu
echo "#define BLKY 8" >> global_mem.cu
echo "#define BLKZ 8" >> global_mem.cu


# Concatenate the remain
cat template_global_mem.cu >> global_mem.cu

# Compile
nvcc -arch=sm_20 global_mem.cu -o test_stencil_3d_global

#Redirect and display
./test_stencil_3d_global >> test_stencil_3d_global.log