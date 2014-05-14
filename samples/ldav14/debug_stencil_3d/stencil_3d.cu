#include "stencil_3d.hpp"
#include "helper_math.h" 
#include <stdio.h>
void stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream);

__global__ 
void __stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo);

void stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream)
{
    dim3 blockDim(32, 32, 1);
	// dim3 gridDim(
        // (dimx/blockDim.x+((dimx%blockDim.x)?1:0)),
        // (dimy/blockDim.y+((dimy%blockDim.y)?1:0)),
        // // (dimz/blockDim.z+((dimz%blockDim.z)?1:0)) );
		// 1); /// Sweep the z dimension, 3D
	dim3 blockSize(64, 64, 1);
    dim3 gridDim(
        (dimx/blockSize.x+((dimx%blockSize.x)?1:0)),
        (dimy/blockSize.y+((dimy%blockSize.y)?1:0)),
        // (dimz/blockDim.z+((dimz%blockDim.z)?1:0)) );
		1); /// Sweep the z dimension, 3D

    // size_t sharedMemSize  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(float);
    // size_t sharedMemSize  = (blockSize.x+2*halo)*(blockSize.y+2*halo)*(blockSize.z+2*halo)*sizeof(float);
    size_t sharedMemSize  = (blockSize.x+2*halo)*(blockSize.y+2*halo)*(blockSize.z+0*halo)*sizeof(float);
    __stencil_3d<<<gridDim, blockDim, sharedMemSize, stream>>>
		(deviceSrc, deviceDst, dimx, dimy, dimz, halo);
}

#define at(x, y, z, dimx, dimy, dimz) ( clamp((int)(z), 0, dimz-1)*dimy*dimx+      \
                                        clamp((int)(y), 0, dimy-1)*dimx+           \
                                        clamp((int)(x), 0, dimx-1) )                   
__global__ 
void __stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo)
{
    extern __shared__ float sharedMem[];                     										
	int3 opened_index_3d, closed_index_3d, offset_index_3d, global_index_3d;
	int  opened_index_1d, closed_index_1d, offset_index_1d, global_index_1d;
	int3 openedDim,  closedDim;
	int  openedSize, closedSize;
	int  thisReading, thisWriting;
	int  numThreads, numReading, numWriting, batch, sweep;
	float result;

	for(sweep=0; sweep<dimz; sweep++)
	{
	
	//Calculate the closed form, instruction parallelism
	closedDim  = make_int3(2*blockDim.x,
	 		 		       2*blockDim.y,
						   1*blockDim.z);
	openedDim  = make_int3(closedDim.x + 2*halo,
	 					   closedDim.y + 2*halo,
						   closedDim.z + 0*halo);
						  
	offset_index_3d  = make_int3(blockIdx.x * closedDim.x, 
								 blockIdx.y * closedDim.y,
								 // blockIdx.z * closedDim.z);
								 sweep * closedDim.z);
	///
	numThreads = blockDim.x  * blockDim.y  * blockDim.z;
	openedSize = openedDim.x * openedDim.y * openedDim.z;
	closedSize = closedDim.x * closedDim.y * closedDim.z;
	
	///
	numReading = (openedSize / numThreads) + ((openedSize % numThreads)?1:0);    
	numWriting = (closedSize / numThreads) + ((closedSize % numThreads)?1:0);    
	
	#pragma unroll
	for(thisReading=0; thisReading<numReading; thisReading++)
	{
		opened_index_1d =  threadIdx.z * blockDim.y * blockDim.x +                      										
						   threadIdx.y * blockDim.x +                                   										
						   threadIdx.x +                  
						   thisReading * numThreads; //Flatten everything
		opened_index_3d = make_int3((opened_index_1d % (openedDim.y*openedDim.x) % openedDim.x),		
								    (opened_index_1d % (openedDim.y*openedDim.x) / openedDim.x),		
									(opened_index_1d / (openedDim.y*openedDim.x)) );  
		global_index_3d = make_int3((offset_index_3d.x + opened_index_3d.x - 1*halo),
									(offset_index_3d.y + opened_index_3d.y - 1*halo),
									(offset_index_3d.z + opened_index_3d.z - 0*halo) );
		global_index_1d = global_index_3d.z * dimy * dimx +
						  global_index_3d.y * dimx +
						  global_index_3d.x;
		if (opened_index_3d.y < openedDim.y)
		{
			if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&	
			   global_index_3d.y >= 0 && global_index_3d.y < dimy &&
			   global_index_3d.x >= 0 && global_index_3d.x < dimx) 
			{
				sharedMem[at(opened_index_3d.x, 
							 opened_index_3d.y, 
							 opened_index_3d.z,
						     openedDim.x, 
							 openedDim.y, 
							 openedDim.z)]
				= deviceSrc[global_index_1d];
				
				//Debug
				// deviceDst[global_index_1d]= deviceSrc[global_index_1d];
			}
		}
	}
	__syncthreads();
	
	
	#pragma unroll
	for(thisWriting=0; thisWriting<numWriting; thisWriting++)
	{
		closed_index_1d =  threadIdx.z * blockDim.y * blockDim.x +                      										
						   threadIdx.y * blockDim.x +                                   										
						   threadIdx.x +                  
						   thisWriting * numThreads; //Magic is here 
		closed_index_3d = make_int3((closed_index_1d % (closedDim.y*closedDim.x) % closedDim.x),		
								    (closed_index_1d % (closedDim.y*closedDim.x) / closedDim.x),		
									(closed_index_1d / (closedDim.y*closedDim.x)) );  
		global_index_3d = make_int3((offset_index_3d.x + closed_index_3d.x),
									(offset_index_3d.y + closed_index_3d.y),
									(offset_index_3d.z + closed_index_3d.z) );
		global_index_1d = global_index_3d.z * dimy * dimx +
						  global_index_3d.y * dimx +
						  global_index_3d.x;
						  
						  
		result	= sharedMem[at(closed_index_3d.x + 1*halo + 0, 
							   closed_index_3d.y + 1*halo + 0, 
							   closed_index_3d.z + 0*halo + 0,
						       openedDim.x, 
							   openedDim.y, 
							   openedDim.z)];
		// if(result !=  deviceSrc[global_index_1d])
		// {
			// printf("Error in %03d %03d\n",  
				// global_index_3d.x, global_index_3d.y, global_index_3d.z);
		// }
							
		if (closed_index_3d.y < closedDim.y)
		{
			if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&	
			   global_index_3d.y >= 0 && global_index_3d.y < dimy &&
			   global_index_3d.x >= 0 && global_index_3d.x < dimx) 
			{
				//Debug
				// deviceDst[global_index_1d]= deviceSrc[global_index_1d];
				deviceDst[global_index_1d] = result;
				// deviceDst[global_index_1d] =
				// sharedMem[at(closed_index_3d.x + 1*halo, 
							 // closed_index_3d.y + 1*halo, 
							 // closed_index_3d.z + 0*halo,
						     // openedDim.x, 
							 // openedDim.y, 
							 // openedDim.z)];
				
				// deviceDst[global_index_1d]= deviceSrc[global_index_1d];
			}
		}
	}
	
	}
}                                                                                         
