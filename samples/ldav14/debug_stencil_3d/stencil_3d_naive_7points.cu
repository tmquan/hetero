#include "stencil_3d.hpp"
#include "helper_math.h" 
#include <stdio.h>

#define DIMX 512
#define DIMY 512
#define DIMZ 512
__forceinline__ __device__ __host__
float getValPtr(cudaPitchedPtr* array, int x, int y, int z)
{
	// return ( (float) (((char*)array->ptr+ z * array->pitch*array->pitch->ysize) + y * array->pitch) + x);
	return *(( (float*) ((  (char*)array->ptr + z*array->pitch*array->ysize) + y*array->pitch) + x));
	// return *(float*)((((char*)array.ptr+ z*array.pitch*DIMY)+ y*array.pitch)+ x);
}

__forceinline__ __device__ __host__
void setValPtr(cudaPitchedPtr* array, int x, int y, int z, float val)
{
	*(( (float*) ((  (char*)array->ptr + z*array->pitch*array->ysize) + y*array->pitch) + x)) = val;
	// *((float*)((((char*)array.ptr + z*array.pitch*DIMY) + y*array.pitch) + x)) = val;
	return;
}

void stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream);

__global__ 
void __stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo);

void stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream)
{
    dim3 blockDim(32, 8, 1);
	dim3 blockSize(32, 8, 1);
	// dim3 blockSize(64, 16, 1);
	// dim3 blockDim(8, 8, 8);
	// dim3 blockSize(8, 8, 8);
    dim3 gridDim(
        (dimx/blockSize.x+((dimx%blockSize.x)?1:0)),
        (dimy/blockSize.y+((dimy%blockSize.y)?1:0)),
        // (dimz/blockSize.z+((dimz%blockSize.z)?1:0)) );
        1);
	size_t sharedMemSize  = (blockSize.x+2*halo)*(blockSize.y+2*halo)*(blockSize.z+2*halo)*sizeof(float);
    __stencil_3d_naive_7points<<<gridDim, blockDim, sharedMemSize, stream>>>
		(deviceSrc, deviceDst, dimx, dimy, dimz, halo);
}

#define at(x, y, z, dimx, dimy, dimz) ( clamp((z), 0, dimz-1)*dimy*dimx+      \
                                        clamp((y), 0, dimy-1)*dimx+           \
                                        clamp((x), 0, dimx-1) )                

__global__ 
void __stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo)
{
	int3 index_3d = make_int3(blockDim.x * blockIdx.x + threadIdx.x,
							  blockDim.y * blockIdx.y + threadIdx.y,
							  blockDim.z * blockIdx.z + threadIdx.z);
	// char* d_src = (char*)deviceSrc.ptr; 
	// char* d_dst = (char*)deviceDst.ptr; 
	// size_t pitch = deviceSrc.pitch; 
	// size_t slicePitch = pitch * dimy;
	// size_t slicePitch = pitch * deviceSrc.ysize;
	
	// char* sliceSrc;
	// char* sliceDst;
	// float* rowSrc;
	// float* rowDst;
	// Debug: Write back to the device Result 1 to 1
	// sliceSrc = d_src + index_3d.z * slicePitch; 	
	// rowSrc = (float*)(sliceSrc + index_3d.y * pitch); 
	// result = rowSrc[index_3d.x];
	// sliceDst = d_dst + index_3d.z * slicePitch; 	
	// rowDst = (float*)(sliceDst + index_3d.y * pitch); 
	// rowDst[index_3d.x] = result;
	// return;
	
	float result, tmp, alpha, beta;
	beta 	= 0.0625f;
	alpha 	= 0.1f;	
	
		

	
	// Debug: Write back to the device Result 1 to 1, inline function
	// 95.0712   GFLOPS/s
	// setValPtr(deviceDst, index_3d.x, index_3d.y, index_3d.z, getValPtr(deviceSrc, index_3d.x, index_3d.y, index_3d.z));
	
	// 147.3512   GFLOPS/s
	// for(int sweepPlane=0; sweepPlane<dimz; sweepPlane++)
	// {
		// index_3d = make_int3(blockDim.x * blockIdx.x + threadIdx.x,
							 // blockDim.y * blockIdx.y + threadIdx.y,
							 // blockDim.z * sweepPlane + threadIdx.z);
		// result = getValPtr(deviceSrc, index_3d.x+0, index_3d.y+0, index_3d.z+0);
		// setValPtr(deviceDst, index_3d.x, index_3d.y, index_3d.z, result);
	// }
	// return;
	
	// Debug: Naive implementation stencil_3d from global mem, no sweepPlane
	//14.7918   GFLOPS/s
	// if(((index_3d.z >0) && (index_3d.z < (dimz-1))) &&
	    // ((index_3d.y >0) && (index_3d.y < (dimy-1))) &&
	    // ((index_3d.x >0) && (index_3d.x < (dimx-1))) )
	// {
		// tmp = getValPtr(deviceSrc, index_3d.x+1, index_3d.y+0, index_3d.z+0) +
			  // getValPtr(deviceSrc, index_3d.x-1, index_3d.y+0, index_3d.z+0) +
			  // getValPtr(deviceSrc, index_3d.x+0, index_3d.y+1, index_3d.z+0) +
			  // getValPtr(deviceSrc, index_3d.x+0, index_3d.y-1, index_3d.z+0) +
			  // getValPtr(deviceSrc, index_3d.x+0, index_3d.y+0, index_3d.z+1) +
			  // getValPtr(deviceSrc, index_3d.x+0, index_3d.y+0, index_3d.z-1);
		// result = alpha*getValPtr(deviceSrc, index_3d.x+0, index_3d.y+0, index_3d.z+0) + beta*tmp;
		// setValPtr(deviceDst, index_3d.x, index_3d.y, index_3d.z, result);
	// }
	// return;
	
	// Debug: Naive implementation stencil_3d from global mem, with sweepPlane
	//  37.1160   GFLOPS/s
	// for(int sweepPlane=0; sweepPlane<dimz; sweepPlane++)
	// {
		// index_3d = make_int3(blockDim.x * blockIdx.x + threadIdx.x,
							 // blockDim.y * blockIdx.y + threadIdx.y,
							 // blockDim.z * sweepPlane + threadIdx.z);
		// if(((index_3d.z >0) && (index_3d.z < (dimz-1))) &&
		   // ((index_3d.y >0) && (index_3d.y < (dimy-1))) &&
		   // ((index_3d.x >0) && (index_3d.x < (dimx-1))) )
		// {
			// tmp = getValPtr(&deviceSrc, index_3d.x+1, index_3d.y+0, index_3d.z+0) +
				  // getValPtr(&deviceSrc, index_3d.x-1, index_3d.y+0, index_3d.z+0) +
				  // getValPtr(&deviceSrc, index_3d.x+0, index_3d.y+1, index_3d.z+0) +
				  // getValPtr(&deviceSrc, index_3d.x+0, index_3d.y-1, index_3d.z+0) +
				  // getValPtr(&deviceSrc, index_3d.x+0, index_3d.y+0, index_3d.z+1) +
				  // getValPtr(&deviceSrc, index_3d.x+0, index_3d.y+0, index_3d.z-1);
			// result = alpha*getValPtr(&deviceSrc, index_3d.x+0, index_3d.y+0, index_3d.z+0) + beta*tmp;
			// setValPtr(&deviceDst, index_3d.x, index_3d.y, index_3d.z, result);
		// }
	// }
	// return;
	
	//ILP along x and y*array, not finish
	int3 opened_index_3d, closed_index_3d, offset_index_3d, global_index_3d;
	int  opened_index_1d, closed_index_1d, offset_index_1d, global_index_1d;
	int3 openedDim,  closedDim;
	int  openedSize, closedSize;
	int  thisReading, thisWriting;
	int  numThreads, numReading, numWriting;
	 
	for(int sweepPlane=0; sweepPlane<dimz; sweepPlane++)
	{
		//Calculate the closed form, instruction parallelism
		closedDim  = make_int3(2*blockDim.x, //Adjust the factor here
							   2*blockDim.y, //Adjust the factor here
							   1*blockDim.z);
		// openedDim  = make_int3(closedDim.x + 2*halo,
							   // closedDim.y + 2*halo,
							   // closedDim.z + 2*halo);
							  
		offset_index_3d  = make_int3(blockIdx.x * closedDim.x, 
									 blockIdx.y * closedDim.y,
									 // blockIdx.z * closedDim.z);
									 sweepPlane * closedDim.z);
		///
		numThreads = blockDim.x  * blockDim.y  * blockDim.z;
		// openedSize = openedDim.x * openedDim.y * openedDim.z;
		closedSize = closedDim.x * closedDim.y * closedDim.z;
		
		///
		// numReading = (openedSize / numThreads) + ((openedSize % numThreads)?1:0);    
		numWriting = (closedSize / numThreads) + ((closedSize % numThreads)?1:0);   
	
		// index_3d = make_int3(blockDim.x * blockIdx.x + threadIdx.x,
							 // blockDim.y * blockIdx.y + threadIdx.y,
							 // blockDim.z * sweepPlane + threadIdx.z);
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
			// global_index_1d = global_index_3d.z * dimy * dimx +
							  // global_index_3d.y * dimx +
							  // global_index_3d.x;
							  
			if(((index_3d.z >0) && (index_3d.z < (dimz-1))) &&
			   ((index_3d.y >0) && (index_3d.y < (dimy-1))) &&
			   ((index_3d.x >0) && (index_3d.x < (dimx-1))) )
			{
				tmp = getValPtr(&deviceSrc, global_index_3d.x+1, global_index_3d.y+0, global_index_3d.z+0) +
					  getValPtr(&deviceSrc, global_index_3d.x-1, global_index_3d.y+0, global_index_3d.z+0) +
					  getValPtr(&deviceSrc, global_index_3d.x+0, global_index_3d.y+1, global_index_3d.z+0) +
					  getValPtr(&deviceSrc, global_index_3d.x+0, global_index_3d.y-1, global_index_3d.z+0) +
					  getValPtr(&deviceSrc, global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z+1) +
					  getValPtr(&deviceSrc, global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z-1);
				result = alpha*getValPtr(&deviceSrc, global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z+0) + beta*tmp;
				setValPtr(&deviceDst, global_index_3d.x, global_index_3d.y, global_index_3d.z, result);
			}
		}
	}
	return;
	
	
	
	
	
	
	
	
	// // Debug: Naive implementation stencil_3d from global mem
	// if(((index_3d.z >0) && (index_3d.z < (dimz-1))) &&
	   // ((index_3d.y >0) && (index_3d.y < (dimy-1))) &&
	   // ((index_3d.x >0) && (index_3d.x < (dimx-1))) )
	// {
		
		// for(int zz=-1; zz<2; zz++)
		// {
			// sliceSrc = d_src + (index_3d.z + zz) * slicePitch; 
			// for(int yy=-1; yy<2; yy++)
			// {
				// rowSrc = (float*)(sliceSrc + (index_3d.y + yy) * pitch); 
				// for(int xx=-1; xx<2; xx++)
				// {
					// if((zz!=0) && (yy!=0) && (xx!=0))
					// {
						// tmp += rowSrc[index_3d.x + xx];
					// }
				// }
			// }
		// }
	// }
	// sliceSrc = d_src + (index_3d.z) * slicePitch; 
	// rowSrc = (float*)(sliceSrc + (index_3d.y) * pitch); 
	// result = alpha*rowSrc[index_3d.x] + beta*tmp;	
	
	// sliceDst = d_dst + index_3d.z * slicePitch; 	
	// rowDst = (float*)(sliceDst + index_3d.y * pitch); 
	// rowDst[index_3d.x] = result;
	// return;

	
	// // Debug: Stencil no shared mem

	
	// // Debug: Stencil with shared mem
	// extern __shared__ float sharedMem[];                     										
	// // __shared__ float sharedMem[34][10][3];                     										
	// int3 opened_index_3d, closed_index_3d, offset_index_3d, global_index_3d;
	// int  opened_index_1d, closed_index_1d, offset_index_1d, global_index_1d;
	// int3 openedDim,  closedDim;
	// int  openedSize, closedSize;
	// int  thisReading, thisWriting;
	// int  numThreads, numReading, numWriting, batch, sweepPlane;
	
	// // for(sweepPlane=0; sweepPlane<dimz; sweepPlane++)
	// // {
	
	// //Calculate the closed form, instruction parallelism
	// closedDim  = make_int3(1*blockDim.x,
	 		 		       // 1*blockDim.y,
						   // 1*blockDim.z);
	// openedDim  = make_int3(closedDim.x + 2*halo,
	 					   // closedDim.y + 2*halo,
						   // closedDim.z + 2*halo);
						  
	// offset_index_3d  = make_int3(blockIdx.x * closedDim.x, 
								 // blockIdx.y * closedDim.y,
								 // blockIdx.z * closedDim.z);
								 // // sweepPlane * closedDim.z);
	// ///
	// numThreads = blockDim.x  * blockDim.y  * blockDim.z;
	// openedSize = openedDim.x * openedDim.y * openedDim.z;
	// closedSize = closedDim.x * closedDim.y * closedDim.z;
	
	// ///
	// numReading = (openedSize / numThreads) + ((openedSize % numThreads)?1:0);    
	// numWriting = (closedSize / numThreads) + ((closedSize % numThreads)?1:0);    
	
	
	// #pragma unroll
	// for(thisReading=0; thisReading<numReading; thisReading++)
	// {
		// opened_index_1d =  threadIdx.z * blockDim.y * blockDim.x +                      										
						   // threadIdx.y * blockDim.x +                                   										
						   // threadIdx.x +                  
						   // thisReading * numThreads; //Flatten everything
		// opened_index_3d = make_int3((opened_index_1d % (openedDim.y*openedDim.x) % openedDim.x),		
								    // (opened_index_1d % (openedDim.y*openedDim.x) / openedDim.x),		
									// (opened_index_1d / (openedDim.y*openedDim.x)) );  
									
		// global_index_3d = make_int3((offset_index_3d.x + opened_index_3d.x - 1*halo),
									// (offset_index_3d.y + opened_index_3d.y - 1*halo),
									// (offset_index_3d.z + opened_index_3d.z - 1*halo) );
		// global_index_1d = global_index_3d.z * dimy * dimx +
						  // global_index_3d.y * dimx +
						  // global_index_3d.x;
		// // if(global_index_1d == 0) printf("numReading (%d), numWriting (%d) \n", numReading, numWriting);
		// if (opened_index_3d.z < openedDim.z)
		// {
			// if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&	
			   // global_index_3d.y >= 0 && global_index_3d.y < dimy &&
			   // global_index_3d.x >= 0 && global_index_3d.x < dimx) 
			// {
				// sliceSrc = d_src + (global_index_3d.z) * slicePitch; 
				// rowSrc = (float*)(sliceSrc + (global_index_3d.y) * pitch); 
				// sharedMem[at(opened_index_3d.x, 
							 // opened_index_3d.y, 
							 // opened_index_3d.z,
						     // openedDim.x, 
							 // openedDim.y, 
							 // openedDim.z)]
				// = rowSrc[global_index_3d.x];
			// }
		// }
		// __syncthreads();	
	// }
	
	
	// #pragma unroll
	// for(thisWriting=0; thisWriting<numWriting; thisWriting++)
	// {
		// closed_index_1d =  threadIdx.z * blockDim.y * blockDim.x +                      										
						   // threadIdx.y * blockDim.x +                                   										
						   // threadIdx.x +                  
						   // thisWriting * numThreads; //Magic is here 
		// closed_index_3d = make_int3((closed_index_1d % (closedDim.y*closedDim.x) % closedDim.x),		
								    // (closed_index_1d % (closedDim.y*closedDim.x) / closedDim.x),		
									// (closed_index_1d / (closedDim.y*closedDim.x)) );  
		// global_index_3d = make_int3((offset_index_3d.x + closed_index_3d.x),
									// (offset_index_3d.y + closed_index_3d.y),
									// (offset_index_3d.z + closed_index_3d.z) );
		// global_index_1d = global_index_3d.z * dimy * dimx +
						  // global_index_3d.y * dimx +
						  // global_index_3d.x;
						  
						  
		// result	= sharedMem[at(closed_index_3d.x + 1*halo + 0, 
							   // closed_index_3d.y + 1*halo + 0, 
							   // closed_index_3d.z + 1*halo + 0,
						       // openedDim.x, 
							   // openedDim.y, 
							   // openedDim.z)];
								
		// if (closed_index_3d.z < closedDim.z)
		// {
			// if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&	
			   // global_index_3d.y >= 0 && global_index_3d.y < dimy &&
			   // global_index_3d.x >= 0 && global_index_3d.x < dimx) 
			// {
				// // deviceDst[global_index_1d] = result;
				// sliceDst = d_dst + closed_index_3d.z * slicePitch; 	
				// rowDst = (float*)(sliceDst + closed_index_3d.y * pitch); 
				// rowDst[closed_index_3d.x] = result;
			// }
		// }
	// }
	
	

	// sliceSrc = d_src + (index_3d.z) * slicePitch; 
	// rowSrc = (float*)(sliceSrc + (index_3d.y) * pitch); 
	// result = alpha*rowSrc[index_3d.x] + beta*tmp;	
	
	
	
	// } //End sweepPlane
	
	// Write back to the device Result
	// sliceDst = d_dst + index_3d.z * slicePitch; 	
	// rowDst = (float*)(sliceDst + index_3d.y * pitch); 
	// rowDst[index_3d.x] = result;
	
	
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
