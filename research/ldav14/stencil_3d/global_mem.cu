
#define BLOCKDIMX 16
#define BLOCKDIMY 4
#define BLOCKDIMZ 1
#define BLOCKSIZEX 16 
#define BLOCKSIZEY 32 
#define BLOCKSIZEZ 32 
// #define BLOCKDIMX 32
// #define BLOCKDIMY 8
// #define BLOCKDIMZ 1

// #define BLOCKSIZEX 64
// #define BLOCKSIZEY 16
// #define BLOCKSIZEZ 512

// Use all constants to debug and get the performance
#define DIMX 512
#define DIMY 512
#define DIMZ 512
#define DIMXY (DIMX*DIMY)
#define TOTAL (DIMX*DIMY*DIMZ)


#define NUMTHREADS (BLOCKDIMX*BLOCKDIMY*BLOCKDIMZ)
#define HALO 1
#define OPENEDDIMX  	(BLOCKSIZEX+2*HALO)
#define OPENEDDIMY  	(BLOCKSIZEY+2*HALO)
#define OPENEDDIMZ  	(BLOCKSIZEZ+2*HALO)
#define OPENEDDIMXY 	(OPENEDDIMX*OPENEDDIMY)
#define OPENEDDIMXYZ  	(OPENEDDIMX*OPENEDDIMY*OPENEDDIMZ)
#define CLOSEDDIMX  	(BLOCKSIZEX)
#define CLOSEDDIMY  	(BLOCKSIZEY)
#define CLOSEDDIMZ  	(BLOCKSIZEZ)
#define CLOSEDDIMXY 	(CLOSEDDIMX*CLOSEDDIMY)
#define CLOSEDDIMXYZ  	(CLOSEDDIMX*CLOSEDDIMY*CLOSEDDIMZ)
#define NUMREADING  	((OPENEDDIMXYZ / NUMTHREADS) + ((OPENEDDIMXYZ%NUMTHREADS)?1:0))
#define NUMWRITING  	((CLOSEDDIMXYZ / NUMTHREADS) + ((CLOSEDDIMXYZ%NUMTHREADS)?1:0))

// #define CORRECTNESS_DATA
#define CORRECTNESS_HEAT
// #define myclamp(x, value, tx, fx) {return ((x)==(value)) ? (tx):(fx)}
#define C0 0.25f
#define C1 0.50f


#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string>
// #include <sys/ioctl.h>
#include <cuda.h>
#include <helper_math.h>
// #include <gpu_timer.hpp>

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
		fs->open(filename, ios::in|ios::binary);									\
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


#define at(x, y, z, DIMX, DIMY, DIMZ) ( clamp((int)(z), 0, DIMZ-1)*DIMY*DIMX +  	\
                                        clamp((int)(y), 0, DIMY-1)*DIMX +       	\
                                        clamp((int)(x), 0, DIMX-1) )                   

////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void heatflow_global(float *src, float *dst)
{
	int  closed_index_1d, offset_index_1d, global_index_1d;
	int3 closed_index_3d, offset_index_3d, global_index_3d;
	offset_index_3d  = make_int3(blockIdx.x * BLOCKSIZEX, 
								 blockIdx.y * BLOCKSIZEY,
								 blockIdx.z * BLOCKSIZEZ);
	int index = threadIdx.z * blockDim.y * blockDim.x +
				threadIdx.y * blockDim.x +                                   										
				threadIdx.x;
				
	#pragma unroll
	for(int thisWriting=0; thisWriting<NUMWRITING; thisWriting++)
	{
		// closed_index_1d = threadIdx.z * blockDim.y * blockDim.x + 	
						  // threadIdx.y * blockDim.x +
						  // threadIdx.x + 		
		closed_index_1d = index +
						  thisWriting*NUMTHREADS;
		closed_index_3d = make_int3((closed_index_1d % CLOSEDDIMXY % CLOSEDDIMX),		
								    (closed_index_1d % CLOSEDDIMXY / CLOSEDDIMX),		
									(closed_index_1d / CLOSEDDIMXY) );  // very expensive
		global_index_3d = make_int3((offset_index_3d.x + closed_index_3d.x),
									(offset_index_3d.y + closed_index_3d.y),
									(offset_index_3d.z + closed_index_3d.z) );
		global_index_1d = global_index_3d.z * DIMY * DIMX +
						  global_index_3d.y * DIMX +
						  global_index_3d.x;
		if(global_index_3d.z > 0 && global_index_3d.z < (DIMZ-1) &&	
		   global_index_3d.y > 0 && global_index_3d.y < (DIMY-1) &&
		   global_index_3d.x > 0 && global_index_3d.x < (DIMX-1) ) 
		{
			// dst[at(global_index_3d.x, global_index_3d.y, global_index_3d.z, DIMX, DIMY, DIMZ)] 
			dst[global_index_1d] 
			= C0 * (src[at(global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z+0, DIMX, DIMY, DIMZ)])+
			  C1 * (src[at(global_index_3d.x-1, global_index_3d.y+0, global_index_3d.z+0, DIMX, DIMY, DIMZ)] +
					src[at(global_index_3d.x+1, global_index_3d.y+0, global_index_3d.z+0, DIMX, DIMY, DIMZ)] +
					src[at(global_index_3d.x+0, global_index_3d.y-1, global_index_3d.z+0, DIMX, DIMY, DIMZ)] +
					src[at(global_index_3d.x+0, global_index_3d.y+1, global_index_3d.z+0, DIMX, DIMY, DIMZ)] +
					src[at(global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z-1, DIMX, DIMY, DIMZ)] +
					src[at(global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z+1, DIMX, DIMY, DIMZ)]);
		}
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void heatflow_shared(float *src, float *dst)
{
	
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	printf("-----------------------------------------------------------------------\n");
	srand(time(NULL)); // for random number generator

	cudaSetDevice(2);checkLastError();
	cudaDeviceReset();checkLastError();
	// Specify dimensions

	// Allocate host memory
	float *h_src = new float[TOTAL];
	float *h_dst = new float[TOTAL];
	
	// Allocate device memory
	float *d_src;
	float *d_dst;

	cudaMalloc((void**)&d_src, TOTAL*sizeof(float));		checkLastError();
	cudaMalloc((void**)&d_dst, TOTAL*sizeof(float));		checkLastError();
	
	// Initialize the image source
	for(int z=0; z<DIMZ; z++)
	{
		for(int y=0; y<DIMY; y++)
		{
			for(int x=0; x<DIMX; x++)
			{
				h_src[z*DIMY*DIMX+y*DIMX+x] = (float)( (int)rand() % 10); // 7;
			}
		}
	}
	
	// Transferring to the device memory
	cudaMemcpy(d_src, h_src, TOTAL*sizeof(float), cudaMemcpyHostToDevice); checkLastError();
	cudaMemset(d_dst, 0, TOTAL*sizeof(float));checkLastError();
	
	// parameters for performance eval
	double flops, gbps, nops, nbp;
	nbp = 8*4; // # of bytes transferred per point
	nops = 8.; // # of flops per point
	int iter = 20;
	int rightData = 0;
	int rightHeat = 0;
	/// Verify the correctness of data
// #ifdef CORRECTNESS_DATA
	cudaMemcpy(d_dst, d_src, TOTAL*sizeof(float), cudaMemcpyDeviceToDevice); checkLastError();
	cudaMemcpy(h_dst, d_dst, TOTAL*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();
	for(int z=0; z<DIMZ && rightData; z++)
	{
		for(int y=0; y<DIMY && rightData; y++)
		{
			for(int x=0; x<DIMX && rightData; x++)
			{
				if(h_src[z*DIMY*DIMX+y*DIMX+x] != h_dst[z*DIMY*DIMX+y*DIMX+x])
				{
					printf("Data does not match at x: %d, y: %d, z: %d\n", x, y, z);
					rightData = 0;
					// goto cleanup_data;
				}
			}
		}
	}
	if(rightData)		printf("Data is correct.\n");
// cleanup_data:
// #endif
	// grid construction
	dim3 numThreads(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ); //Dim
	dim3 numBlocks((DIMX/BLOCKSIZEX)+((DIMX%BLOCKSIZEX)?1:0),	//Size  for ILP
				   (DIMY/BLOCKSIZEY)+((DIMY%BLOCKSIZEY)?1:0),
				   (DIMZ/BLOCKSIZEZ)+((DIMZ%BLOCKSIZEZ)?1:0));
	cudaMemset(d_dst, 0, TOTAL*sizeof(float));checkLastError(); // Reset the result
	memset(h_dst, 0, TOTAL*sizeof(float));
	printf("Blockdim (%03d, %03d, %03d); Blocksize (%03d, %03d, %03d);\n",
		BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ, BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
	// launch kernel
	// GpuTimer gpu_timer;
	// gpu_timer.Start();
	cudaEvent_t begin, end;

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaEventRecord(begin, 0);
	for(int n=0; n<iter; n++)
	{
		heatflow_global<<<numBlocks, numThreads>>>(d_src, d_dst);
	}
	// gpu_timer.Stop();
	cudaDeviceSynchronize();
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	float msec;
	cudaEventElapsedTime(&msec, begin, end);
	checkLastError();
	
	// float msec = gpu_timer.Elapsed();
	gbps = nbp*DIMX*DIMY*DIMZ/(msec/1000.)/(1024.*1024.*1024.)*(double)iter;
	flops = nops*DIMX*DIMY*DIMZ/(msec/1000.)/(1024.*1024.*1024.)*(double)iter;
	printf("Computing time : %.3f msec, Device memory bandwidth : %.3f GB/s, GFLOPS : %.3f\n", 		msec, gbps, flops);
	

	float* h_ref = new float[DIMX*DIMY*DIMZ];
	float tmp, result;
// #ifdef CORRECTNESS_HEAT
	/// Verify the correctness of heat flow, no check at boundary
	// Golden result
	for(int z=1; z<(DIMZ-1); z++)
	{
		for(int y=1; y<(DIMY-1); y++)
		{
			for(int x=1; x<(DIMX-1); x++)
			{
				result = C0 * (h_src[at(x+0, y+0, z+0, DIMX, DIMY, DIMZ)])+
						 C1 * (h_src[at(x-1, y+0, z+0, DIMX, DIMY, DIMZ)] +
							   h_src[at(x+1, y+0, z+0, DIMX, DIMY, DIMZ)] +
							   h_src[at(x+0, y-1, z+0, DIMX, DIMY, DIMZ)] +
							   h_src[at(x+0, y+1, z+0, DIMX, DIMY, DIMZ)] +
							   h_src[at(x+0, y+0, z-1, DIMX, DIMY, DIMZ)] +
							   h_src[at(x+0, y+0, z+1, DIMX, DIMY, DIMZ)]);		
				h_ref[at(x+0, y+0, z+0, DIMX, DIMY, DIMZ)] 	= result;
			}
		}
	} 
	// Transferring to the host memory
	cudaMemcpy(h_dst, d_dst, TOTAL*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();
	// Compare result
	for(int z=1; z<(DIMZ-1) && rightHeat; z++)
	{
		for(int y=1; y<(DIMY-1) && rightHeat; y++)
		{
			for(int x=1; x<(DIMX-1) && rightHeat; x++)
			{
				if(h_ref[z*DIMY*DIMX+y*DIMX+x] != h_dst[z*DIMY*DIMX+y*DIMX+x])
				{
					printf("Solution does not match at x: %d, y: %d, z: %d\n", x, y, z);
					printf("h_ref (%04.4f), h_dst (%04.4f)\n", 
						h_ref[z*DIMY*DIMX+y*DIMX+x], 
						h_dst[z*DIMY*DIMX+y*DIMX+x]);
					rightHeat = 0;
					// goto cleanup_heat;
				}
			}
		}
	}
	if(rightHeat)	printf("Solution is correct.\n");
// cleanup_heat:
// #endif
	///!!! Print line
	// struct winsize w;
    // ioctl(0, TIOCGWINSZ, &w);
	// for(int k=0; k<w.ws_col; k++) 
		// printf("-");
	printf("\n");
	checkLastError();	
// cleanup:
	cudaFree(d_src);
	cudaFree(d_dst);
	free(h_src);
	free(h_dst);
	free(h_ref);
	return 0;
}