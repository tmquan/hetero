#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
using namespace std;

// ----------------------------------------------------------------------------
#define checkLastError() {                                          				\
	cudaError_t error = cudaGetLastError();                               			\
	int id; 																		\
	cudaGetDevice(&id);																\
	if(error != cudaSuccess) {                                         				\
		printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",	\
			__FILE__,__LINE__, cudaGetErrorString(error), id);			      	 	\
		exit(EXIT_FAILURE);  														\
	}                                                               				\
}
// ---------------------------------------------------------------------------- 
__global__
void __convolution(float *src, float *dst, int dimx, int dimy, int dimz)
{
	__shared__ float sharedMem[14][14][14];

	int  shared_index_1d, global_index_1d, index_1d;
	// int2 shared_index_2d, global_index_2d, index_2d;
	int3 shared_index_3d, global_index_3d, index_3d;
	
	// Multi batch loading
	int trial;
	for(trial=0; trial <6; trial++)
	{
		shared_index_1d 	= threadIdx.z * blockDim.y * blockDim.x +
							  threadIdx.y * blockDim.x + 
							  threadIdx.x +
							  blockDim.x  * blockDim.y * blockDim.z * trial;  // Next number of loading
		shared_index_3d		= make_int3((shared_index_1d % ((blockDim.y+2*3) * (blockDim.x+2*3))) % (blockDim.x+2*3),
										(shared_index_1d % ((blockDim.y+2*3) * (blockDim.x+2*3))) / (blockDim.x+2*3),
										(shared_index_1d / ((blockDim.y+2*3) * (blockDim.x+2*3))) );
		global_index_3d		= make_int3(blockIdx.x * blockDim.x + shared_index_3d.x - 3,
										blockIdx.y * blockDim.y + shared_index_3d.y - 3,
										blockIdx.z * blockDim.z + shared_index_3d.z - 3);
		global_index_1d 	= global_index_3d.z * dimy * dimx + 
							  global_index_3d.y * dimx + 
							  global_index_3d.x;
		if (shared_index_3d.z < (blockDim.z + 2*3)) 
		{
			if (global_index_3d.z >= 0 && global_index_3d.z < dimz && 
				global_index_3d.y >= 0 && global_index_3d.y < dimy &&
				global_index_3d.x >= 0 && global_index_3d.x < dimx )	
				sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = src[global_index_1d];
			else
				sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = -1;
		}
		__syncthreads();
	}
	// // First batch loading
	// shared_index_1d 	= threadIdx.z * blockDim.y * blockDim.x +
						  // threadIdx.y * blockDim.x + 
						  // threadIdx.x;
	// shared_index_3d		= make_int3((shared_index_1d % ((blockDim.y+2*3) * (blockDim.x+2*3))) % (blockDim.x+2*3),
									// (shared_index_1d % ((blockDim.y+2*3) * (blockDim.x+2*3))) / (blockDim.x+2*3),
									// (shared_index_1d / ((blockDim.y+2*3) * (blockDim.x+2*3))) );
	// global_index_3d		= make_int3(blockIdx.x * blockDim.x + shared_index_3d.x - 3,
									// blockIdx.y * blockDim.y + shared_index_3d.y - 3,
									// blockIdx.z * blockDim.z + shared_index_3d.z - 3);
	// global_index_1d 	= global_index_3d.z * dimy * dimx + 
						  // global_index_3d.y * dimx + 
						  // global_index_3d.x;
	// if (global_index_3d.z >= 0 && global_index_3d.z < dimz && 
		// global_index_3d.y >= 0 && global_index_3d.y < dimy &&
		// global_index_3d.x >= 0 && global_index_3d.x < dimx )	
		// sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = src[global_index_1d];
	// else
		// sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = -1;
	// __syncthreads();
	
	// // Second batch loading
	// shared_index_1d 	= threadIdx.z * blockDim.y * blockDim.x +
						  // threadIdx.y * blockDim.x + 
						  // threadIdx.x +
						  // blockDim.x  * blockDim.y * blockDim.z;  // Next number of loading
	// shared_index_3d		= make_int3((shared_index_1d % ((blockDim.y+2*3) * (blockDim.x+2*3))) % (blockDim.x+2*3),
									// (shared_index_1d % ((blockDim.y+2*3) * (blockDim.x+2*3))) / (blockDim.x+2*3),
									// (shared_index_1d / ((blockDim.y+2*3) * (blockDim.x+2*3))) );
	// global_index_3d		= make_int3(blockIdx.x * blockDim.x + shared_index_3d.x - 3,
									// blockIdx.y * blockDim.y + shared_index_3d.y - 3,
									// blockIdx.z * blockDim.z + shared_index_3d.z - 3);
	// global_index_1d 	= global_index_3d.z * dimy * dimx + 
						  // global_index_3d.y * dimx + 
						  // global_index_3d.x;
	// if (shared_index_3d.z < (blockDim.z + 2*3)) 
	// {
		// if (global_index_3d.z >= 0 && global_index_3d.z < dimz && 
			// global_index_3d.y >= 0 && global_index_3d.y < dimy &&
			// global_index_3d.x >= 0 && global_index_3d.x < dimx )	
			// sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = src[global_index_1d];
		// else
			// sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = -1;
	// }
	// __syncthreads();

	index_3d		= make_int3(blockIdx.x * blockDim.x + threadIdx.x,
								blockIdx.y * blockDim.y + threadIdx.y,
								blockIdx.z * blockDim.z + threadIdx.z);
	index_1d 		= index_3d.z * dimy * dimx + 
					  index_3d.y * dimx + 
					  index_3d.x;
	
	// Store back
	if (index_3d.z < dimz && 
		index_3d.y < dimy && 
		index_3d.x < dimx)
		dst[index_1d] = sharedMem[threadIdx.z+3][threadIdx.y+3][threadIdx.x+3];
}
void convolution(float *src, float *dst, int dimx, int dimy, int dimz)
{
	dim3 numBlocks((dimx/8 + ((dimx%8)?1:0)),
				   (dimy/8 + ((dimy%8)?1:0)),
				   (dimz/8 + ((dimz%8)?1:0)));
	dim3 numThreads(8, 8, 8);
	__convolution<<<numBlocks, numThreads>>>(src, dst, dimx, dimy, dimz);
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
	srand(time(NULL)); // for random number generator
	// Specify dimensions
	const int dimx  = 100;
	const int dimy  = 100;
	const int dimz  = 100;

	const int total = dimx*dimy*dimz;
	
	// Allocate host memory
	float *h_src = new float[total];
	float *h_dst = new float[total];
	
	// Allocate device memory
	float *d_src;
	float *d_dst;
	
	cudaMalloc((void**)&d_src, total*sizeof(float));		checkLastError();
	cudaMalloc((void**)&d_dst, total*sizeof(float));		checkLastError();
	
	// Initialize the image source
	for(int z=0; z<dimz; z++)
	{
		for(int y=0; y<dimy; y++)
		{
			for(int x=0; x<dimx; x++)
			{
				h_src[z*dimy*dimx+y*dimx+x] = (float)rand();
			}
		}
	}
	// Transferring to the device memory
	cudaMemcpy(d_src, h_src, total*sizeof(float), cudaMemcpyHostToDevice); checkLastError();
	
	convolution(d_src, d_dst, dimx, dimy, dimz);

	cudaMemcpy(h_dst, d_dst, total*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();
	
	// Verify the result
	for(int z=0; z<dimz; z++)
	{
		for(int y=0; y<dimy; y++)
		{
			for(int x=0; x<dimx; x++)
			{
				if(h_src[z*dimy*dimx+y*dimx+x] != h_dst[z*dimy*dimx+y*dimx+x])
				{
					printf("Solution doesnot match at x: %d, y: %d, z: %d\n", x, y, z);
					goto cleanup;
				}
				// else
					// printf("Solution match at x: %d, y: %d, z: %d\n", x, y, z);
			}
		}
	}
	printf("Solution is correct.\n");
cleanup:
	cudaFree(d_src);
	cudaFree(d_dst);
	free(h_src);
	free(h_dst);
	return 0;
}
