#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string>
#include <hetero_cmdparser.hpp>
#include <gpu_timer.hpp>
#include <cuda.h>
#include "stencil_3d07.hpp"
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
#define checkWriteFile(filename, pData, size) {                    					\
		fstream *fs = new fstream;													\
		fs->open(filename, ios::out|ios::binary);									\
		if (!fs->is_open())															\
		{																			\
			fprintf(stderr, "Cannot open file '%s' in file '%s' at line %i\n",		\
			filename, __FILE__, __LINE__);											\
			return 1;																\
		}																			\
		fs->write(reinterpret_cast<char*>(pData), size);							\
		fs->close();																\
		delete fs;																	\
	}
	
////////////////////////////////////////////////////////////////////////////////////////////////////
// extern __constant__ float alpha;
// extern __constant__ float beta;
////////////////////////////////////////////////////////////////////////////////////////////////////
const char* key =
	"{ h   |help    |      | print help message }"	
	"{ dx  |dimx    | 512  | dimensionx }"
	"{ dy  |dimy    | 512  | dimensiony }"
	"{ dz  |dimz    | 512  | dimensionz }"
	"{ bx  |blockx  | 4    | blockDimx }"
	"{ by  |blocky  | 4    | blockDimy }"
	"{ bz  |blockz  | 4    | blockDimz }"
	"{ ilp |istrlp  | 1    | instruction parallelism factor }"
	"{ num |num     | 20   | numLoops }"
	;
////////////////////////////////////////////////////////////////////////////////////////////////////
// extern texture<float, 1, cudaReadModeElementType> tex;         // 3D texture
////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	srand(time(NULL)); // for random number generator
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	cudaSetDevice((int)rand()%numDevices);
	cudaSetDevice(0);
	cudaDeviceReset();
	// Specify dimensions
	// Parsing the arguments
	CommandLineParser cmd(argc, argv, key);
	const int numTrials			= cmd.get<int>("num", false);
	const int dimx  			= cmd.get<int>("dimx", false);
	const int dimy  			= cmd.get<int>("dimy", false);
	const int dimz  			= cmd.get<int>("dimz", false);

	const int total = dimx*dimy*dimz;
	
	const int bx  			= cmd.get<int>("bx", false);
	const int by  			= cmd.get<int>("by", false);
	const int bz  			= cmd.get<int>("bz", false);
	
	const int ilp  			= cmd.get<int>("ilp", false);
	
	cmd.printParams();
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
	// float a = -6.0f;
	// float b = +0.1f;
	// cudaBindTexture(NULL, tex, d_src, total*sizeof(float));checkLastError();
	// cudaMemcpyToSymbol(alpha, &a, sizeof(float), 0, cudaMemcpyHostToDevice);
	// cudaMemcpyToSymbol(beta, &b, sizeof(float), 0, cudaMemcpyHostToDevice);
	GpuTimer gpu_timer;
	gpu_timer.Start();
	for(int n=0; n<numTrials; n++)
		stencil_3d07(d_src, d_dst, dimx, dimy, dimz, bx, by, bz, ilp, 1);
	gpu_timer.Stop();
	
	float ms = gpu_timer.Elapsed()/numTrials;
	printf("Time %4.3f ms\n", ms);	
	int numOps = 8;
	float gflops = (float)total*(float)numOps* 1.0e-9f/(ms*1.0e-3f);
	printf("Performance of %s with blockx (%02d), blocky (%02d), blockz (%02d), ilp factor (%02d) is %04.4f   GFLOPS/s\n", argv[0], bx, by, bz, ilp, gflops); 
	checkLastError();
	  // Compute and print the performance
    // float msecPerMatrixMul = msecTotal / nIter;
    // double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    // printf(
        // "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        // gigaFlops,
        // msecPerMatrixMul,
        // flopsPerMatrixMul,
        // threads.x * threads.y);
	
	cudaMemcpy(h_dst, d_dst, total*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();
	
	cudaFree(d_src);
	cudaFree(d_dst);
	free(h_src);
	free(h_dst);
	return 0;
}