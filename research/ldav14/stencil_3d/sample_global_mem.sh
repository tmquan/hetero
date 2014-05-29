#!/bin/bash

let BLOCKDIMX=32
let BLOCKDIMY=8
let BLOCKDIMZ=1
 
let BLOCKSIZEX=64
let BLOCKSIZEY=16
let BLOCKSIZEZ=512

echo $str
# Create a new source code

# touch global_mem.cu
# rm global_mem.cu
# rm test_stencil_3d_global

# Construct the code
echo "" > global_mem.cu
echo "#define BLOCKDIMX $BLOCKDIMX" >> global_mem.cu
echo "#define BLOCKDIMY $BLOCKDIMY" >> global_mem.cu
echo "#define BLOCKDIMZ $BLOCKDIMZ" >> global_mem.cu

echo "#define BLOCKSIZEX $BLOCKSIZEX" >> global_mem.cu
echo "#define BLOCKSIZEY $BLOCKSIZEY" >> global_mem.cu
echo "#define BLOCKSIZEZ $BLOCKSIZEZ" >> global_mem.cu


# Concatenate the remain
cat template_global_mem.cu >> global_mem.cu

# Compile
nvcc -arch=sm_20 global_mem.cu --disable-warnings -Xcompiler -Wextra -o test_stencil_3d_global

#Redirect and display
./test_stencil_3d_global #>> test_stencil_3d_global.log
			