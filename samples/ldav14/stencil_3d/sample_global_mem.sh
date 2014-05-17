#!/bin/bash

let bx=8
let by=8
let bz=8
let ilp=8
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

# #Redirect and display
./test_stencil_3d_global #>> test_stencil_3d_global.log
			