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
void __convolution(float *src, float *dst, int dimx, int dimy)
{
	__shared__ float sharedMem[12][12];

	int  shared_index_1d, global_index_1d, index_1d;
	int2 shared_index_2d, global_index_2d, index_2d;
	
	// First batch loading
	shared_index_1d 	= threadIdx.y * blockDim.x + threadIdx.x;
	shared_index_2d		= make_int2(shared_index_1d % 12,
									shared_index_1d / 12);
	global_index_2d		= make_int2(blockIdx.x * blockDim.x + shared_index_2d.x - 2,
									blockIdx.y * blockDim.y + shared_index_2d.y - 2);
	global_index_1d 	= global_index_2d.y * dimx + global_index_2d.x;
	if (global_index_2d.y >= 0 && global_index_2d.y < dimy && 
		global_index_2d.x >= 0 && global_index_2d.x < dimx)	
		sharedMem[shared_index_2d.y][shared_index_2d.x] = src[global_index_1d];
	else
		sharedMem[shared_index_2d.y][shared_index_2d.x] = -1;
	__syncthreads();
	
	// Second batch loading
	shared_index_1d 	= threadIdx.y * blockDim.x + threadIdx.x 
						+ blockDim.x * blockDim.y;
	shared_index_2d		= make_int2(shared_index_1d % 12,
									shared_index_1d / 12);
	global_index_2d		= make_int2(blockIdx.x * blockDim.x + shared_index_2d.x - 2,
									blockIdx.y * blockDim.y + shared_index_2d.y - 2);
	global_index_1d 	= global_index_2d.y * dimx + global_index_2d.x;
	if (shared_index_2d.y < 12) 
	{
		if (global_index_2d.y >= 0 && global_index_2d.y < dimy && 
			global_index_2d.x >= 0 && global_index_2d.x < dimx)
			sharedMem[shared_index_2d.y][shared_index_2d.x] = src[global_index_1d];
		else
			sharedMem[shared_index_2d.y][shared_index_2d.x] = -1;
	}
	__syncthreads();

	index_2d		= make_int2(blockIdx.x * blockDim.x + threadIdx.x,
								blockIdx.y * blockDim.y + threadIdx.y);
	index_1d 		= index_2d.y * dimx + index_2d.x;
	
	// Store back
	if (index_2d.y < dimy && index_2d.x < dimx)
		dst[index_1d] = sharedMem[threadIdx.y+2][threadIdx.x+2];
}
void convolution(float *src, float *dst, int dimx, int dimy)
{
	dim3 numBlocks((dimx/8 + ((dimx%8)?1:0)),
				   (dimy/8 + ((dimy%8)?1:0)));
	dim3 numThreads(8, 8);
	__convolution<<<numBlocks, numThreads>>>(src, dst, dimx, dimy);
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
	srand(time(NULL)); // for random number generator
	// Specify dimensions
	const int dimx  = 100;
	const int dimy  = 100;

	const int total = dimx*dimy;
	
	// Allocate host memory
	float *h_src = new float[total];
	float *h_dst = new float[total];
	
	// Allocate device memory
	float *d_src;
	float *d_dst;
	
	cudaMalloc((void**)&d_src, total*sizeof(float));		checkLastError();
	cudaMalloc((void**)&d_dst, total*sizeof(float));		checkLastError();
	
	// Initialize the image source
	for(int y=0; y<dimy; y++)
	{
		for(int x=0; x<dimx; x++)
		{
			h_src[y*dimx+x] = (float)rand();
		}
	}

	// Transferring to the device memory
	cudaMemcpy(d_src, h_src, total*sizeof(float), cudaMemcpyHostToDevice); checkLastError();
	
	convolution(d_src, d_dst, dimx, dimy);

	cudaMemcpy(h_dst, d_dst, total*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();
	
	// Verify the result
	for(int y=0; y<dimy; y++)
	{
		for(int x=0; x<dimx; x++)
		{
			if(h_src[y*dimx+x] != h_dst[y*dimx+x])
			{
				printf("Solution doesnot match at x: %d, y: %d\n", x, y);
				goto cleanup;
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
