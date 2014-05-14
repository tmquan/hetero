#include "stencil_3d.hpp"
#include "helper_math.h" 
#include <stdio.h>
void stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream);

__global__ 
void __stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo);

void stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream)
{
    dim3 blockDim(8, 8, 8);
	dim3 blockSize(8, 8, 8);
    dim3 gridDim(
        (dimx/blockSize.x+((dimx%blockSize.x)?1:0)),
        (dimy/blockSize.y+((dimy%blockSize.y)?1:0)),
        (dimz/blockSize.z+((dimz%blockSize.z)?1:0)) );

    __stencil_3d_naive_7points<<<gridDim, blockDim, 0, stream>>>
		(deviceSrc, deviceDst, dimx, dimy, dimz, halo);
}

#define at(x, y, z, dimx, dimy, dimz) ( clamp((int)(z), 0, dimz-1)*dimy*dimx+      \
                                        clamp((int)(y), 0, dimy-1)*dimx+           \
                                        clamp((int)(x), 0, dimx-1) )                   
__global__ 
void __stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo)
{
	int3 index_3d = make_int3(blockDim.x * blockIdx.x + threadIdx.x,
							  blockDim.y * blockIdx.y + threadIdx.y,
							  blockDim.z * blockIdx.z + threadIdx.z);
	/// Naive copy
	deviceDst[index_3d.z][index_3d.y][index_3d.x] 
	= deviceSrc[index_3d.z][index_3d.y][index_3d.x];
}                                                                                         
