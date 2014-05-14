#include "stencil_3d.hpp"
#include "helper_math.h" 
#include <stdio.h>

#define DIMX 512
#define DIMY 512
#define DIMZ 512

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

// #define at(x, y, z, dimx, dimy, dimz) ( clamp((int)(z), 0, dimz-1)*dimy*dimx+      \
                                        // clamp((int)(y), 0, dimy-1)*dimx+           \
                                        // clamp((int)(x), 0, dimx-1) )                
// #define at(ptr, x, y, z										
__global__ 
void __stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo)
{
	int3 index_3d = make_int3(blockDim.x * blockIdx.x + threadIdx.x,
							  blockDim.y * blockIdx.y + threadIdx.y,
							  blockDim.z * blockIdx.z + threadIdx.z);
	char* d_src = (char*)deviceSrc.ptr; 
	char* d_dst = (char*)deviceDst.ptr; 
	size_t pitch = deviceSrc.pitch; 
	size_t slicePitch = pitch * dimy;
	
	char* sliceSrc;
	char* sliceDst;
	float* rowSrc;
	float* rowDst;
	
	float result, tmp, alpha, beta;
	beta = 0.0625f;
	alpha = 0.1f;	
	if(((index_3d.z >0) && (index_3d.z < (dimz-1))) &&
	   ((index_3d.y >0) && (index_3d.y < (dimy-1))) &&
	   ((index_3d.x >0) && (index_3d.x < (dimx-1))) )
	{
		
		for(int zz=-1; zz<2; zz++)
		{
			sliceSrc = d_src + (index_3d.z + zz) * slicePitch; 
			for(int yy=-1; yy<2; yy++)
			{
				rowSrc = (float*)(sliceSrc + (index_3d.y + yy) * pitch); 
				for(int xx=-1; xx<2; xx++)
				{
					if((zz!=0) && (yy!=0) && (xx!=0))
					{
						tmp += rowSrc[index_3d.x + xx];
					}
				}
			}
		}
	}

	sliceSrc = d_src + (index_3d.z) * slicePitch; 
	rowSrc = (float*)(sliceSrc + (index_3d.y) * pitch); 
	result = alpha*rowSrc[index_3d.x] + beta*tmp;	
	
	// Write back to the device Result
	sliceDst = d_dst + index_3d.z * slicePitch; 	
	rowDst = (float*)(sliceDst + index_3d.y * pitch); 
	rowDst[index_3d.x] = result;
	
	
	
	// /// Naive copy
	// // d_dst[index_3d.z][index_3d.y][index_3d.x] 
	// // = d_src[index_3d.z][index_3d.y][index_3d.x];
	// char* d_src = (char*)deviceSrc.ptr; 
	// char* d_dst = (char*)deviceDst.ptr; 
	// size_t pitch = deviceSrc.pitch; 
	// size_t slicePitch = pitch * dimy;
	// // for (int z = 0; z < dimz; ++z) 
	// // { 
		// char* sliceSrc = d_src + index_3d.z * slicePitch; 
		// char* sliceDst = d_dst + index_3d.z * slicePitch; 
		// // for (int y = 0; y < dimy; ++y) 
		// // { 
			// float* rowSrc = (float*)(sliceSrc + index_3d.y * pitch); 
			// float* rowDst = (float*)(sliceDst + index_3d.y * pitch); 
			// // for (int x = 0; x < dimx; ++x) 
			// // { 
				// rowDst[index_3d.x] = rowSrc[index_3d.x];
			// // } 
		// // } 
	// // }
}                                                                                         
