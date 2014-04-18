#include "stencil_2d.hpp"
#include "helper_math.h" 

void stencil_2d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int halo, cudaStream_t stream);

__global__ 
void __stencil_2d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int halo);

void stencil_2d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int halo, cudaStream_t stream)
{
    dim3 blockDim(8, 8);
    dim3 gridDim(
        (dimx/blockDim.x + ((dimx%blockDim.x)?1:0)),
        (dimy/blockDim.y + ((dimy%blockDim.y)?1:0)) );
    size_t sharedMemSize  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*sizeof(float);
    __stencil_2d<<<gridDim, blockDim, sharedMemSize, stream>>>
     (deviceSrc, deviceDst, dimx, dimy, halo);
}

#define at(x, y, dimx, dimy) ( clamp(y, 0, dimy-1)*dimx +     \
                               clamp(x, 0, dimx-1) )            
__global__ 
void __stencil_2d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int halo)
{
    extern __shared__ float sharedMemSrc[];                     	
    int  shared_index_1d, global_index_1d, index_1d;                                      	
    int2 shared_index_2d, global_index_2d, index_2d;                                      	
    // Multi batch reading here                                                           	
    int2 sharedMemDim    = make_int2(blockDim.x+2*halo,                                   	
                                     blockDim.y+2*halo);                                  	
    int  sharedMemSize   = sharedMemDim.x*sharedMemDim.y;                                 	
    int2 blockSizeDim    = make_int2(blockDim.x+0*halo,                                   	
                                     blockDim.y+0*halo);                                  	
    int blockSize        = blockSizeDim.x*blockSizeDim.y;                                 	
    int numBatches       = sharedMemSize/blockSize + ((sharedMemSize%blockSize)?1:0);     	
    for(int batch=0; batch<numBatches; batch++)                                           	
    {                                                                                     	
        shared_index_1d  =  threadIdx.y * blockDim.x +                                    	
                            threadIdx.x +                                                 	
                            blockSize*batch; //Magic is here quantm@unist.ac.kr           	
        shared_index_2d  =  make_int2(shared_index_1d % (blockDim.x+2*halo),              	
                                      shared_index_1d / (blockDim.x+2*halo) );            	
        global_index_2d  =  make_int2(blockIdx.x * blockDim.x + shared_index_2d.x - halo, 	
                                      blockIdx.y * blockDim.y + shared_index_2d.y - halo);	
        global_index_1d  =  global_index_2d.y * dimx +                                    	
                            global_index_2d.x;                                            	
        if (shared_index_2d.y < (blockDim.y + 2*halo))                                    	
        {                                                                                 	
            if(global_index_2d.y >= 0 && global_index_2d.y < dimy &&                      	
               global_index_2d.x >= 0 && global_index_2d.x < dimx)                        	
            {                                                                             	
                sharedMemSrc[at(shared_index_2d.x, shared_index_2d.y, sharedMemDim.x, sharedMemDim.y)] = deviceSrc[global_index_1d];                         
            }                                                                             
            else                                                                          
            {                                                                             
                sharedMemSrc[at(shared_index_2d.x, shared_index_2d.y, sharedMemDim.x, sharedMemDim.y)] = -1;                                                     
            }                                                                             
        }                                                                                 
        __syncthreads();                                                                  
    }                                                                                     
                                                                                          
    // Stencil  processing here                                                           
    float result = sharedMemSrc[at(shared_index_2d.x + halo, shared_index_2d.y + halo, sharedMemDim.x, sharedMemDim.y)];                         
	                                                                                       
    // Single pass writing here                                                           
    index_2d       =  make_int2(blockIdx.x * blockDim.x + threadIdx.x,                    
                                blockIdx.y * blockDim.y + threadIdx.y);                   
    index_1d       =  index_2d.y * dimx +                                                 
                      index_2d.x;                                                         
	                                                                                       
    if (index_2d.y < dimy &&                                                              
        index_2d.x < dimx)                                                                
    {                                                                                     
        deviceDst[index_1d] = result;                                        
    }                                                                                     
}                                                                                         
