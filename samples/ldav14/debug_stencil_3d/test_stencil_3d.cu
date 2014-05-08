#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string>
#include <sys/ioctl.h>
#include <cuda.h>
#include <gpu_timer.hpp>
#include <hetero_cmdparser.hpp>
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
////////////////////////////////////////////////////////////////////////////////////////////////////


#define at(x, y, z, dimx, dimy, dimz) ( clamp((int)(z), 0, dimz-1)*dimy*dimx +       \
                                        clamp((int)(y), 0, dimy-1)*dimx +            \
                                        clamp((int)(x), 0, dimx-1) )                   

////////////////////////////////////////////////////////////////////////////////////////////////////
const char* key =
	"{ h   |help    |      | print help message }"	
	"{ dx  |dimx    | 512  | dimensionx }"
	"{ dy  |dimy    | 512  | dimensiony }"
	"{ dz  |dimz    | 512  | dimensionz }"
	"{ bx  |blockx  | 4    | blockDimx }"
	"{ by  |blocky  | 4    | blockDimy }"
	"{ bz  |blockz  | 1    | blockDimz }"
	"{ ilp |istrlp  | 1    | instruction parallelism factor }"
	"{ num |num     | 20   | numLoops }"
	;
////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	srand(time(NULL)); // for random number generator

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

	cudaSetDevice(2);checkLastError();
	cudaDeviceReset();checkLastError();
	// Specify dimensions

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




	GpuTimer gpu_timer;
	gpu_timer.Start();
	for(int n=0; n<numTrials; n++)
		// stencil_3d07(d_src, d_dst, dimx, dimy, dimz, bx, by, bz, ilp, 1);
		stencil_3d(d_src, d_dst, dimx, dimy, dimz, 1);
	gpu_timer.Stop();

	float ms = gpu_timer.Elapsed()/numTrials;
	printf("Time %4.3f ms\n", ms);	


	int numOps = 30;
	float gflops = (float)total*(float)numOps* 1.0e-9f/(ms*1.0e-3f);
	printf("Performance of %s is %04.4f   GFLOPS/s\n", argv[0],  gflops); 

	// Transferring to the host memory
	cudaMemcpy(h_dst, d_dst, total*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();

	///!!! Verify the result
	for(int z=0; z<dimz; z++)
	{
		for(int y=0; y<dimy; y++)
		{
			for(int x=0; x<dimx; x++)
			{
				if(h_src[z*dimy*dimx+y*dimx+x] != h_dst[z*dimy*dimx+y*dimx+x])
				{
					printf("Solution does not match at x: %d, y: %d, z: %d\n", x, y, z);
					goto cleanup;
				}
			}
		}
	}
	printf("Solution is correct.\n");
	
	///!!! Print line
	struct winsize w;
    ioctl(0, TIOCGWINSZ, &w);
	for(int k=0; k<w.ws_col; k++) 
		printf("-");
	printf("\n");
	checkLastError();	
cleanup:
	cudaFree(d_src);
	cudaFree(d_dst);
	free(h_src);
	free(h_dst);
	return 0;
}