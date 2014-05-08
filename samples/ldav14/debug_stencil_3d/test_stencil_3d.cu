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
__global__ 
void __copy_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo)
{
	
	// Single pass writing here                                                           
	int3 index_3d       =  make_int3(blockIdx.x * blockDim.x + threadIdx.x,                    
								blockIdx.y * blockDim.y + threadIdx.y,                    
								blockIdx.z * blockDim.z + threadIdx.z);                   
	int index_1d       =  index_3d.z * dimy * dimx +                                          
					  index_3d.y * dimx +                                                 
					  index_3d.x;                                                         
																						   
	if (index_3d.z < dimz &&                                                              
		index_3d.y < dimy &&                                                              
		index_3d.x < dimx)                                                                
	{                                                                                     
		deviceDst[index_1d] = deviceSrc[index_1d];                                        
	} 


}  
void copy_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream)
{
    dim3 blockDim(32, 4, 4);
    dim3 gridDim(
        (dimx/blockDim.x + ((dimx%blockDim.x)?1:0)),
        (dimy/blockDim.y + ((dimy%blockDim.y)?1:0)),
        (dimz/blockDim.z + ((dimz%blockDim.z)?1:0)) );
		// 1); /// Sweep the z dimension, 3D
    // size_t sharedMemSize  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(float);
    __copy_3d<<<gridDim, blockDim, 0, stream>>>
     (deviceSrc, deviceDst, dimx, dimy, dimz, halo);
}                                                                                       

////////////////////////////////////////////////////////////////////////////////////////////////////
texture<float, 3, cudaReadModeElementType> tex; 
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
	
	cudaSetDevice(0);
	cudaDeviceReset();
	// Specify dimensions
	
	// Allocate host memory
	float *h_src = new float[total];
	float *h_dst = new float[total];
	
	// Allocate device memory
	// float *d_src;
	// float *d_dst;
	// cudaMalloc((void**)&d_src, total*sizeof(float));		checkLastError();
	// cudaMalloc((void**)&d_dst, total*sizeof(float));		checkLastError();
	
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
	
		
	///!!! Setting coefficients here
	// float a = -6.0f;
  	// float b = +0.1f;
	
	///!!! Setting texture parameter here
	// cudaExtent volumeSize = make_cudaExtent(dimx, dimy, dimz);

	tex.normalized = false;      //Donot normalize to [0, 1]
	
	// cudaFilterModePoint 	 	 Point filter mode
	// cudaFilterModeLinear 	 Linear filter mode
    tex.filterMode = cudaFilterModePoint;      // linear interpolation

	// cudaAddressModeWrap 	 	 Wrapping address mode
	// cudaAddressModeClamp 	 Clamp to edge address mode
	// cudaAddressModeMirror 	 Mirror address mode
	// cudaAddressModeBorder 	 Border address mode
	tex.addressMode[0] = cudaAddressModeMirror;  
    tex.addressMode[1] = cudaAddressModeMirror ;
    tex.addressMode[2] = cudaAddressModeMirror ;
	
	
	
	///!!! Allocate and copy to device memory
	cudaExtent volumeSize = make_cudaExtent(dimx, dimy, dimz);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();	checkLastError();
	cudaArray *d_src;
	cudaMalloc3DArray(&d_src, &channelDesc, volumeSize);				checkLastError();
	// cudaArray *d_dst;
	// cudaMalloc3DArray(&d_src, &channelDesc, volumeSize);				checkLastError();
	float *d_dst;
	cudaMalloc((void**)&d_dst, total*sizeof(float));		checkLastError();
	
	
	cout << __FILE__ << " " << __LINE__ << endl;
	
	///!!! Transferring to the device memory
	// cudaMemcpy(d_src, h_src, total*sizeof(float), cudaMemcpyHostToDevice); checkLastError();
	cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_src, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_src;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);	checkLastError();


	///!!! Bind the texture memory
	cudaBindTextureToArray(tex, d_src, channelDesc); checkLastError();
	
	

	
	
	GpuTimer gpu_timer;
	gpu_timer.Start();
	for(int n=0; n<numTrials; n++)
		stencil_3d(d_src, d_dst, dimx, dimy, dimz, 1);
	gpu_timer.Stop();
	
	///!!! Normalize the running time
	float ms = gpu_timer.Elapsed()/numTrials; 
	printf("Time %4.3f ms\n", ms);	

	int numOperations, GFLOPS;
	numOperations = 8;
	GFLOPS 		  = (float)total*(float)numOperations* 1.0e-9f/(ms*1.0e-3f);
	printf("Performance of %s is %04.4f   GFLOPS/s\n", argv[0],  GFLOPS); 
	
	
	///!!! Check correctness here
	// cudaMemcpy(h_dst, d_dst, total*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();
	
	///!!! Pring a line to terminate
	struct winsize w;
    ioctl(0, TIOCGWINSZ, &w);
	for(int k=0; k<w.ws_col; k++) 
		printf("-");
	printf("\n");
	checkLastError();

	
	
	// // Verify the result
	// for(int z=0; z<dimz; z++)
	// {
		// for(int y=0; y<dimy; y++)
		// {
			// for(int x=0; x<dimx; x++)
			// {
				// if(h_src[z*dimy*dimx+y*dimx+x] != h_dst[z*dimy*dimx+y*dimx+x])
				// {
					// printf("Solution doesnot match at x: %d, y: %d, z: %d\n", x, y, z);
					// goto cleanup;
				// }
			// }
		// }
	// }
	// printf("Solution is correct.\n");
// cleanup:
	cudaFree(d_src);
	cudaFree(d_dst);
	free(h_src);
	free(h_dst);
	return 0;
}