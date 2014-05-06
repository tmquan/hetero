#include "stencil_3d.hpp"
#include "helper_math.h" 

void stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream);

__global__ 
void __stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo);

void stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream)
{
    dim3 blockDim(32, 8, 1);
    dim3 gridDim(
        (dimx/blockDim.x + ((dimx%blockDim.x)?1:0)),
        (dimy/blockDim.y + ((dimy%blockDim.y)?1:0)),
        // (dimz/blockDim.z + ((dimz%blockDim.z)?1:0)) );
		1); /// Sweep the z dimension, 3D
    size_t sharedMemSize  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(float);
    __stencil_3d<<<gridDim, blockDim, sharedMemSize, stream>>>
     (deviceSrc, deviceDst, dimx, dimy, dimz, halo);
}

#define at(x, y, z, dimx, dimy, dimz) ( clamp((int)z, 0, dimz-1)*dimy*dimx +       \
                                        clamp((int)y, 0, dimy-1)*dimx +            \
                                        clamp((int)x, 0, dimx-1) )                   
__global__ 
void __stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo)
{
    extern __shared__ float sharedMemSrc[];                     										
    int  shared_index_1d, global_index_1d, index_1d;                                      										
    int3 shared_index_3d, global_index_3d, index_3d;                                      										
    // Multi batch reading here                                                           										
    int3 sharedMemDim    = make_int3(blockDim.x+2*halo,                                   										
                                     blockDim.y+2*halo,                                  										
                                     blockDim.z+2*halo);                                  										
    int  sharedMemSize   = sharedMemDim.x*sharedMemDim.y*sharedMemDim.z;                  										
    int3 blockSizeDim    = make_int3(blockDim.x+0*halo,                                   										
                                     blockDim.y+0*halo,                                   										
                                     blockDim.z+0*halo);                                  										
    int  blockSize        = blockSizeDim.x*blockSizeDim.y*blockSizeDim.z;                  									
    int  numBatches       = sharedMemSize/blockSize + ((sharedMemSize%blockSize)?1:0);    

	float result;
	int batch, pass, h;
	//First pass will load entire 3 planes, process and write
	{
		for(batch=0; batch<numBatches; batch++)                                           										
		{                                                                                     										
			shared_index_1d  =  threadIdx.z * blockDim.y * blockDim.x +                       										
								threadIdx.y * blockDim.x +                                    										
								threadIdx.x +                                                 										
								blockSize*batch; //Magic is here quantm@unist.ac.kr           										
			shared_index_3d  =  make_int3((shared_index_1d % ((blockDim.y+2*halo)*(blockDim.x+2*halo))) % (blockDim.x+2*halo),		
										  (shared_index_1d % ((blockDim.y+2*halo)*(blockDim.x+2*halo))) / (blockDim.x+2*halo),		
										  (shared_index_1d / ((blockDim.y+2*halo)*(blockDim.x+2*halo))) );      					
			global_index_3d  =  make_int3(blockIdx.x * blockDim.x + shared_index_3d.x - halo, 										
										  blockIdx.y * blockDim.y + shared_index_3d.y - halo, 										
										  blockIdx.z * blockDim.z + shared_index_3d.z - halo);										
			global_index_1d  =  global_index_3d.z * dimy * dimx +                                    								
								global_index_3d.y * dimx +                                    										
								global_index_3d.x;                                            										
			if (shared_index_3d.z < (blockDim.z + 2*halo))                                    										
			{                                                                                 										
				if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&                      										
				   global_index_3d.y >= 0 && global_index_3d.y < dimy &&                        									
				   global_index_3d.x >= 0 && global_index_3d.x < dimx)                        										
				{                                                                             										
					sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = deviceSrc[global_index_1d];                         
				}                                                                             
			}                                                                                 
			__syncthreads();                                                                  
		}                                                                                     
																							  
		// Stencil  processing here                                                           
		result = sharedMemSrc[at(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];                         
																							   
		// Single pass writing here                                                           
		index_3d       =  make_int3(blockIdx.x * blockDim.x + threadIdx.x,                    
									blockIdx.y * blockDim.y + threadIdx.y,                    
									blockIdx.z * blockDim.z + threadIdx.z);                   
		index_1d       =  index_3d.z * dimy * dimx +                                          
						  index_3d.y * dimx +                                                 
						  index_3d.x;                                                         
																							   
		if (index_3d.z < dimz &&                                                              
			index_3d.y < dimy &&                                                              
			index_3d.x < dimx)                                                                
		{                                                                                     
			deviceDst[index_1d] = result;                                        
		} 
	}
	
	//Second pass, swap the shared memory: middle -> top, bottom -> middle, load the bottom plane, this one iterate from 1 to dimz-1
	for(pass=1; pass<dimz; pass++)
	{
		// break;
		for(batch=0; batch<numBatches; batch++)                                           										
		{                                                                                     										
			shared_index_1d  =  threadIdx.z * blockDim.y * blockDim.x +                       										
								threadIdx.y * blockDim.x +                                    										
								threadIdx.x +                                                 										
								blockSize*batch; //Magic is here quantm@unist.ac.kr           										
			shared_index_3d  =  make_int3((shared_index_1d % ((blockDim.y+2*halo)*(blockDim.x+2*halo))) % (blockDim.x+2*halo),		
										  (shared_index_1d % ((blockDim.y+2*halo)*(blockDim.x+2*halo))) / (blockDim.x+2*halo),		
										  (shared_index_1d / ((blockDim.y+2*halo)*(blockDim.x+2*halo))) );      					
			global_index_3d  =  make_int3(blockIdx.x * blockDim.x + shared_index_3d.x - halo, 										
										  blockIdx.y * blockDim.y + shared_index_3d.y - halo, 										
										  pass       * blockDim.z + shared_index_3d.z - halo);										
			global_index_1d  =  global_index_3d.z * dimy * dimx +                                    								
								global_index_3d.y * dimx +                                    										
								global_index_3d.x;  
			// if ((shared_index_3d.x == 0))	
			// {
				// for(h=0; h<=halo; h++)
				// {		
					// //swap(&sharedMemSrc[h*(sharedMemDim.x*sharedMemDim.y)], &sharedMemSrc[(h+halo)*(sharedMemDim.x*sharedMemDim.y)]);
					// (sharedMemSrc+h*(sharedMemDim.x*sharedMemDim.y) ) = &sharedMemSrc[(h+halo)*(sharedMemDim.x*sharedMemDim.y)];
				// }	
			// }
			if (shared_index_3d.z <= halo)
			{
				// sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = 
				// sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];
			}
				
			if ((shared_index_3d.z < (blockDim.z + 2*halo)) &&  
				(shared_index_3d.z > halo))
			{                                                                                 										
				if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&                      										
				   global_index_3d.y >= 0 && global_index_3d.y < dimy &&                        									
				   global_index_3d.x >= 0 && global_index_3d.x < dimx)                        										
				{                                                                             										
					sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = deviceSrc[global_index_1d];                         
				}                                                    						                                                 
			}                                                                                 
			__syncthreads();                                                                  
		}                                                                                     
																							  
		// Stencil  processing here                                                           
		result = sharedMemSrc[at(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];                         
																							   
		// Single pass writing here                                                           
		index_3d       =  make_int3(blockIdx.x * blockDim.x + threadIdx.x,                    
									blockIdx.y * blockDim.y + threadIdx.y,                    
									pass       * blockDim.z + threadIdx.z);                   
		index_1d       =  index_3d.z * dimy * dimx +                                          
						  index_3d.y * dimx +                                                 
						  index_3d.x;                                                         
																							   
		if (index_3d.z < dimz &&                                                              
			index_3d.y < dimy &&                                                              
			index_3d.x < dimx)                                                                
		{                                                                                     
			deviceDst[index_1d] = result;                                        
		} 
	}
}                                                                                         