#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string>

#include <cuda.h>
#include "stencil_3d.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////////////////////////
#define checkWriteFile(filename, pData, size) {                    				\
		fstream *fs = new fstream;												\
		fs->open(filename, ios::out|ios::binary);								\
		if (!fs->is_open())														\
		{																		\
			fprintf(stderr, "Cannot open file '%s' in file '%s' at line %i\n",	\
			filename, __FILE__, __LINE__);										\
			return 1;															\
		}																		\
		fs->write(reinterpret_cast<char*>(pData), size);						\
		fs->close();															\
		delete fs;																\
	}
int main(int argc, char **argv)
{
srand(time(NULL)); // for random number generator
	// Specify dimensions
	const int dimx  = 600;
	const int dimy  = 600;
	const int dimz  = 600;

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
	
	stencil_3d(d_src, d_dst, dimx, dimy, dimz, 5);

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