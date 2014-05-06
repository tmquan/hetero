#include "stencil_3d07.hpp"
#include "helper_math.h" 
#include "helper_cuda.h" 
texture<float, 1, cudaReadModeElementType> tex;         // 3D texture
void stencil_3d07(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz,  int blockx, int blocky, int blockz, int ilp, int halo, cudaStream_t stream);

__global__ 
void __stencil_3d07(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int ilp, int halo);

void stencil_3d07(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz,  int blockx, int blocky, int blockz, int ilp, int halo, cudaStream_t stream)
{
	cudaBindTexture(NULL, tex, deviceSrc, dimx*dimy*dimz*sizeof(float));
    dim3 blockDim(blockx, blocky, blockz);
    dim3 gridDim(
        (dimx/blockDim.x + ((dimx%blockDim.x)?1:0)),
        (dimy/blockDim.y + ((dimy%blockDim.y)?1:0)),
        (dimz/blockDim.z + ((dimz%blockDim.z)?1:0)) );
    size_t sharedMemSize  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(float);
    __stencil_3d07<<<gridDim, blockDim, sharedMemSize, stream>>>
     (deviceSrc, deviceDst, dimx, dimy, dimz, ilp, halo);
}

#define at(x, y, z, dimx, dimy, dimz) ( clamp((int)(z), 0, dimz-1)*dimy*dimx +       \
                                        clamp((int)(y), 0, dimy-1)*dimx +            \
                                        clamp((int)(x), 0, dimx-1) )                   
__global__ 
void __stencil_3d07(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int ilp, int halo)
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
	
    for(int batch=0; batch<numBatches; batch++)                                           										
    {  
		if(threadIdx.y<(blockDim.y/ilp))
		{
			#pragma unroll
			for(int i=0; i<ilp; i++)
			{
				shared_index_1d  =  threadIdx.z * blockDim.y * blockDim.x +                       										
									(threadIdx.y + i * (blockDim.y/ilp)) * blockDim.x +                                    										
									threadIdx.x  +                                               										
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
						// sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = deviceSrc[global_index_1d];  
						sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = tex1Dfetch(tex, global_index_1d);   
					}                                                                             						
					else                                                                          						
					{                                                                             						
						sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = -1;                                                     
					}                                                                             
				}  
			}
		}	                                                                 
    }                                                                                     
    __syncthreads();                       
	float alpha = -6.0f;
	float beta  = +0.1f;
    // Stencil  processing here                                                           
    // float result = sharedMemSrc[at(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];                  
	float tmp    =  beta*(sharedMemSrc[at(threadIdx.x + halo +1, threadIdx.y + halo+0, threadIdx.z + halo+0, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] +
						  sharedMemSrc[at(threadIdx.x + halo -1, threadIdx.y + halo+0, threadIdx.z + halo+0, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] +
						  sharedMemSrc[at(threadIdx.x + halo +0, threadIdx.y + halo+1, threadIdx.z + halo+0, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] +
						  sharedMemSrc[at(threadIdx.x + halo +0, threadIdx.y + halo-1, threadIdx.z + halo+0, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] +
						  sharedMemSrc[at(threadIdx.x + halo +0, threadIdx.y + halo+0, threadIdx.z + halo+1, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] +
						  sharedMemSrc[at(threadIdx.x + halo +0, threadIdx.y + halo+0, threadIdx.z + halo-1, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)]);
	float result = alpha*sharedMemSrc[at(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] + tmp;                                                                                    
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
