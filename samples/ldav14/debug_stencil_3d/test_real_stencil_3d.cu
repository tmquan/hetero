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
#include "helper_math.h"

using namespace std;
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

#define checkReadFile(filename, pData, size) {                    					\
		fstream *fs = new fstream;													\
		fs->open(filename, ios::in|ios::binary);							\
		if (!fs->is_open())															\
		{																			\
			printf("Cannot open file '%s' in file '%s' at line %i\n",				\
			filename, __FILE__, __LINE__);											\
			return 1;																\
		}																			\
		fs->read(reinterpret_cast<char*>(pData), size);								\
		fs->close();																\
		delete fs;																	\
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
	
	// ----------------------------------------------------------------------------------------
	// Allocate device memory
	// float *d_src;
	// float *d_dst;

	cudaPitchedPtr d_pitchSrc;
	cudaPitchedPtr d_pitchDst;
	// ----------------------------------------------------------------------------------------	
	// w 	- Width in bytes
	// h 	- Height in elements
	// d 	- Depth in elements
	cudaExtent extent = make_cudaExtent(dimx*sizeof(float), dimy, dimz);
	cudaMemcpy3DParms copyParams = {0};

	// ----------------------------------------------------------------------------------------

	cudaMalloc3D(&d_pitchSrc, extent); checkLastError();
	cudaMalloc3D(&d_pitchDst, extent); checkLastError();
	// ----------------------------------------------------------------------------------------

	copyParams.srcPtr.ptr 	= h_src;
	copyParams.srcPtr.pitch = dimx*sizeof(float);
	copyParams.srcPtr.xsize	= dimx;
	copyParams.srcPtr.ysize	= dimy;
	copyParams.dstPtr.ptr   = d_pitchSrc.ptr;
	copyParams.dstPtr.pitch = d_pitchSrc.pitch;
	copyParams.dstPtr.xsize	= dimx;
	copyParams.dstPtr.ysize	= dimy;
	copyParams.extent.width = dimx*sizeof(float);
	copyParams.extent.height= dimx;
	copyParams.extent.depth	= dimy;
	copyParams.kind			= cudaMemcpyHostToDevice;
	
	cudaMemcpy3D(&copyParams);
	checkLastError();
	// ----------------------------------------------------------------------------------------
	// Debug: Direct copy, will be comment out later
	// copyParams.srcPtr.ptr 	= d_pitchSrc.ptr;
	// copyParams.srcPtr.pitch = d_pitchSrc.pitch;
	// copyParams.srcPtr.xsize	= dimx;
	// copyParams.srcPtr.ysize	= dimy;
	// copyParams.dstPtr.ptr   = d_pitchDst.ptr;
	// copyParams.dstPtr.pitch = d_pitchDst.pitch;
	// copyParams.dstPtr.xsize	= dimx;
	// copyParams.dstPtr.ysize	= dimy;
	// copyParams.extent.width = dimx*sizeof(float);
	// copyParams.extent.height= dimx;
	// copyParams.extent.depth	= dimy;
	// copyParams.kind			= cudaMemcpyDeviceToDevice;
	
	// cudaMemcpy3D(&copyParams);
	
	stencil_3d_naive_7points(d_pitchSrc, d_pitchDst, dimx, dimy, dimz, 1);
	checkLastError();
	// ----------------------------------------------------------------------------------------

	
	copyParams.srcPtr.ptr   = d_pitchDst.ptr;
	copyParams.srcPtr.pitch = d_pitchDst.pitch;
	copyParams.srcPtr.xsize	= dimx;
	copyParams.srcPtr.ysize	= dimy;
	copyParams.dstPtr.ptr 	= h_dst;
	copyParams.dstPtr.pitch = dimx*sizeof(float);
	copyParams.dstPtr.xsize	= dimx;
	copyParams.dstPtr.ysize	= dimy;

	copyParams.extent.width = dimx*sizeof(float);
	copyParams.extent.height= dimx;
	copyParams.extent.depth	= dimy;
	copyParams.kind			= cudaMemcpyDeviceToHost;
	
	cudaMemcpy3D(&copyParams);
	
	checkLastError();
	// ----------------------------------------------------------------------------------------
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
	// ----------------------------------------------------------------------------------------
	checkLastError();	
cleanup:
	free(h_src);
	free(h_dst);
	return 0;
	// // checkReadFile("../../../../data/barbara_512x512x512.raw", h_src, total*sizeof(float));
	// // Transferring to the device memory
	// cudaMemcpy(d_src, h_src, total*sizeof(float), cudaMemcpyHostToDevice); checkLastError();
	// cudaMemset(d_dst, 0, total*sizeof(float));checkLastError();



	// GpuTimer gpu_timer;
	// gpu_timer.Start();
	// for(int n=0; n<numTrials; n++)
		// // stencil_3d07(d_src, d_dst, dimx, dimy, dimz, bx, by, bz, ilp, 1);
		// // stencil_3d(d_src, d_dst, dimx, dimy, dimz, 1);
		// stencil_3d_naive_7points(d_src, d_dst, dimx, dimy, dimz, 1);
	// gpu_timer.Stop();

	// float ms = gpu_timer.Elapsed()/numTrials;
	// printf("Time %4.3f ms\n", ms);	


	// int numOps = 8;
	// float gflops = (float)total*(float)numOps* 1.0e-9f/(ms*1.0e-3f);
	// printf("Performance of %s is %04.4f   GFLOPS/s\n", argv[0],  gflops); 

	// // Transferring to the host memory
	// cudaMemcpy(h_dst, d_dst, total*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();

	// ///!!! Verify the result
	// printf("Verifying the result\n");
	// float *h_ref = new float[total];
	// float alpha = 0.01f;
	// float beta = 2.5f;
	// float tmp, result;
	// for(int z=0; z<dimz; z++)
	// {
		// for(int y=0; y<dimy; y++)
		// {
			// for(int x=0; x<dimx; x++)
			// {
				// tmp = beta *    ( h_src[at(x + 1, y + 0, z + 0, dimx, dimy, dimz)] +
								  // h_src[at(x - 1, y + 0, z + 0, dimx, dimy, dimz)] +
								  // h_src[at(x + 0, y + 1, z + 0, dimx, dimy, dimz)] +
								  // h_src[at(x + 0, y - 1, z + 0, dimx, dimy, dimz)] +
								  // h_src[at(x + 0, y + 0, z + 1, dimx, dimy, dimz)] +
								  // h_src[at(x + 0, y + 0, z - 1, dimx, dimy, dimz)] );
				// result = alpha*h_src[at(x + 0, y + 0, z + 0, dimx, dimy, dimz)]  + tmp;			
				// h_ref[at(x, y, z, dimx, dimy, dimz)] 
				// = result;
			// }
		// }
	// }
	
	// for(int z=0; z<dimz; z++)
	// {
		// for(int y=0; y<dimy; y++)
		// {
			// for(int x=0; x<dimx; x++)
			// {
				// if(h_src[z*dimy*dimx+y*dimx+x] != h_dst[z*dimy*dimx+y*dimx+x])
				// {
					// printf("Solution does not match at x: %d, y: %d, z: %d\n", x, y, z);
					// goto cleanup;
				// }
			// }
		// }
	// }
	// printf("Solution is correct.\n");
	
	///!!! Print line
	// struct winsize w;
    // ioctl(0, TIOCGWINSZ, &w);
	// for(int k=0; k<w.ws_col; k++) 
		// printf("-");
	// printf("\n");
	// checkLastError();	
// cleanup:
	// cudaFree(d_src);
	// cudaFree(d_dst);
	// free(h_src);
	// free(h_dst);
	// return 0;
}