#!/bin/bash

# Compile
# nvcc -arch=sm_20 shared_mem.cu --disable-warnings -Xcompiler -Wextra -o test_stencil_3d_shared

echo " " > test_stencil_3d_shared.log
for (( bdz=4; bdz<=512; bdz*=2 )) #Tuning block dim z
do
	for (( bdy=4; bdy<=512; bdy*=2 )) #Tuning block dim y
	do
		for (( bdx=16; bdx<=512; bdx*=2 )) #Tuning block dim x, start from halfwarp
		do
			for (( bsz=4; bsz<=512; bsz*=2 ))
			do
				for (( bsy=4; bsy<=512; bsy*=2 ))
				do
					for (( bsx=bdx; bsx<=512; bsx*=2 ))  #Tuning block size z
					do

						str=$" bdx=$bdx, bdy=$bdy, bdz=$bdz; bsx=$bsx, bsy=$bsy, bsz=$bsz"
						echo $str

						# Construct the code
						echo "" > shared_mem.cu
						echo "#define BLOCKDIMX $bdx" >> shared_mem.cu
						echo "#define BLOCKDIMY $bdy" >> shared_mem.cu
						echo "#define BLOCKDIMZ $bdz" >> shared_mem.cu

						echo "#define BLOCKSIZEX $bsx ">> shared_mem.cu
						echo "#define BLOCKSIZEY $bsy ">> shared_mem.cu
						echo "#define BLOCKSIZEZ $bsz ">> shared_mem.cu


						# Concatenate the remain
						cat template_shared_mem.cu >> shared_mem.cu

						# Compile
						nvcc -arch=sm_20 shared_mem.cu -Xptxas -dlcm=cg --disable-warnings -Xcompiler -Wextra -o test_stencil_3d_shared
						./test_stencil_3d_shared >> test_stencil_3d_shared.log
					done
				done
			done
		done
	done
done
#Redirect and display
# ./test_stencil_3d_shared #>> test_stencil_3d_shared.log
			
			
# echo " " > test_stencil_3d_shared.log
# # printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' - > test_stencil_3d_shared.log
# # for (( ilp=1; ilp<=512; ilp*=2 ))
# for (( ilp=8; ilp<=16; ilp*=2 ))
# do
	# for (( bz=1; bz<=1; bz+=1 ))
	# do
		# for (( bx=1; bx<=512; bx*=2 ))
		# do
			# for (( by=512/bx; by<=512/bx; by*=2 ))
			# do
				# str=$"
				 # bx=$bx, by=$by, bz=$bz, ilp=$ilp"
				# echo $str
				# # Create a new source code
				
				# # touch shared_mem.cu
				# # rm shared_mem.cu
				# # rm test_stencil_3d_shared
				
				# # Construct the code
				# echo "" > shared_mem.cu
				# echo "#define DIMX 512" >> shared_mem.cu
				# echo "#define DIMY 512" >> shared_mem.cu
				# echo "#define DIMZ 512" >> shared_mem.cu

				# echo "#define ILP  $ilp" >> shared_mem.cu


				# echo "#define BLKX $bx" >> shared_mem.cu
				# echo "#define BLKY $by" >> shared_mem.cu
				# echo "#define BLKZ $bz" >> shared_mem.cu
				
				
				# # Concatenate the remain
				# cat template_shared_mem.cu >> shared_mem.cu

				# # Compile
				# nvcc -arch=sm_20 shared_mem.cu -o test_stencil_3d_shared

				# #Redirect and display
				# ./test_stencil_3d_shared >> test_stencil_3d_shared.log
			# done
		# done
	# done
# done

# echo "" > shared_mem.cu
# echo "#define DIMX 512" >> shared_mem.cu
# echo "#define DIMY 512" >> shared_mem.cu
# echo "#define DIMZ 512" >> shared_mem.cu

# echo "#define ILP  8" >> shared_mem.cu


# echo "#define BLKX 8" >> shared_mem.cu
# echo "#define BLKY 8" >> shared_mem.cu
# echo "#define BLKZ 8" >> shared_mem.cu


# # Concatenate the remain
# cat template_shared_mem.cu >> shared_mem.cu

# # Compile
# nvcc -arch=sm_20 shared_mem.cu -o test_stencil_3d_shared

# #Redirect and display
# ./test_stencil_3d_shared >> test_stencil_3d_shared.log