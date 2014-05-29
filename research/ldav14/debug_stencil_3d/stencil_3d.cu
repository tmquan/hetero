#include "stencil_3d.hpp"
#include "helper_math.h" 
#include <stdio.h>

__global__ void stencil_3d(float *src, float *dst, int dimx, int dimy, int dimz, int slice, int ilp, int halo,
	float C0, float C1)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;


	int offset = dimx*y + x;
	//int idx = slice*z + offset;

	int nx, ny, nz, idx, t;
	float center, left, right, top, bottom, front, back;

#pragma unroll 
	for(int i=0; i<ilp; i++)
	{
		//idx += slice*i;
		idx = slice*(ilp*z+i) + offset;
		
		center = src[idx];

		int t = (x == 0) ? 0 : -1;
		left = src[idx + t];
	
		t = (x == dimx-1) ? 0 : 1;
		right = src[idx + t];

		t = (y == 0) ? 0 : -dimx;
		top = src[idx + t];

		t = (y == dimy-1) ? 0 : dimx;
		bottom = src[idx + t];

		t = (z == 0) ? 0 : -(dimx*dimy);
		front = src[idx + t];

		// t = (z == dimz-1) ? 0 : (dimx*dimy);
		t = ((ilp*z + i) == (dimz-1)) ? 0 : (dimx*dimy);
		back = src[idx + t];
		
		dst[idx] = C0*center + C1*(left + right + top + bottom + front + back);
	}
}


// void stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream);
// __global__ 
// void __stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int ilp, int halo);
// void stencil_3d_global(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int ilp, int halo, cudaStream_t stream);
// __global__ 
// void __stencil_3d_global(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo);
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// #define at(x, y, z, dimx, dimy, dimz) ( clamp((int)(z), 0, dimz-1)*dimy*dimx+      \
                                        // clamp((int)(y), 0, dimy-1)*dimx+           \
                                        // clamp((int)(x), 0, dimx-1) )   
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void stencil_3d_global(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int ilp, int halo, cudaStream_t stream)
// {
	// // dim3 dimB(512, 1, 1);
	// // dim3 dimG(
// }
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// void stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo, cudaStream_t stream)
// {
    // // dim3 blockDim(32, 32, 1);
    // dim3 blockDim(8, 8, 8);
	// // dim3 gridDim(
        // // (dimx/blockDim.x+((dimx%blockDim.x)?1:0)),
        // // (dimy/blockDim.y+((dimy%blockDim.y)?1:0)),
        // // // (dimz/blockDim.z+((dimz%blockDim.z)?1:0)) );
		// // 1); /// Sweep the z dimension, 3D
	// // dim3 blockSize(64, 64, 1);
	// dim3 blockSize(8, 8, 8);
    // dim3 gridDim(
        // (dimx/blockSize.x+((dimx%blockSize.x)?1:0)),
        // (dimy/blockSize.y+((dimy%blockSize.y)?1:0)),
        // (dimz/blockDim.z+((dimz%blockDim.z)?1:0)) );
		// // 1); /// Sweep the z dimension, 3D

    // // size_t sharedMemSize  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(float);
    // // size_t sharedMemSize  = (blockSize.x+2*halo)*(blockSize.y+2*halo)*(blockSize.z+2*halo)*sizeof(float);
    // size_t sharedMemSize  = (blockSize.x+2*halo)*(blockSize.y+2*halo)*(blockSize.z+0*halo)*sizeof(float);
    // __stencil_3d<<<gridDim, blockDim, sharedMemSize, stream>>>
		// (deviceSrc, deviceDst, dimx, dimy, dimz, halo);
// }

                
// __global__ 
// void __stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo)
// {
    // extern __shared__ float sharedMem[];                     										
	// int3 opened_index_3d, closed_index_3d, offset_index_3d, global_index_3d;
	// int  opened_index_1d, closed_index_1d, offset_index_1d, global_index_1d;
	// int3 openedDim,  closedDim;
	// int  openedSize, closedSize;
	// int  thisReading, thisWriting;
	// int  numThreads, numReading, numWriting, batch, sweep;
	// // float result;

	// float result, tmp, alpha, beta;
	// beta = 0.0625f;
	// alpha = 0.1f;	
	
	// // Debug: Write back to the device Result 1 to 1
	// // for(sweep=0; sweep<dimz; sweep++)
	// // {
		// // //Calculate the closed form, instruction parallelism
		// // closedDim  = make_int3(2*blockDim.x,
							   // // 2*blockDim.y,
							   // // 1*blockDim.z);
		// // openedDim  = make_int3(closedDim.x + 2*halo,
							   // // closedDim.y + 2*halo,
							   // // closedDim.z + 0*halo);
							  
		// // offset_index_3d  = make_int3(blockIdx.x * closedDim.x, 
									 // // blockIdx.y * closedDim.y,
									 // // // blockIdx.z * closedDim.z);
									 // // sweep * closedDim.z);
		// // ///
		// // numThreads = blockDim.x  * blockDim.y  * blockDim.z;
		// // openedSize = openedDim.x * openedDim.y * openedDim.z;
		// // closedSize = closedDim.x * closedDim.y * closedDim.z;
		
		// // ///
		// // numReading = (openedSize / numThreads) + ((openedSize % numThreads)?1:0);    
		// // numWriting = (closedSize / numThreads) + ((closedSize % numThreads)?1:0);  
		
		
		// // #pragma unroll
		// // for(thisWriting=0; thisWriting<numWriting; thisWriting++)
		// // {
			// // closed_index_1d =  threadIdx.z * blockDim.y * blockDim.x +                      										
							   // // threadIdx.y * blockDim.x +                                   										
							   // // threadIdx.x +                  
							   // // thisWriting * numThreads; //Magic is here 
			// // closed_index_3d = make_int3((closed_index_1d % (closedDim.y*closedDim.x) % closedDim.x),		
										// // (closed_index_1d % (closedDim.y*closedDim.x) / closedDim.x),		
										// // (closed_index_1d / (closedDim.y*closedDim.x)) );  
			// // global_index_3d = make_int3((offset_index_3d.x + closed_index_3d.x),
										// // (offset_index_3d.y + closed_index_3d.y),
										// // (offset_index_3d.z + closed_index_3d.z) );
			// // global_index_1d = global_index_3d.z * dimy * dimx +
							  // // global_index_3d.y * dimx +
							  // // global_index_3d.x;
							  
							  
			// // // result	= sharedMem[at(closed_index_3d.x + 1*halo + 0, 
								   // // // closed_index_3d.y + 1*halo + 0, 
								   // // // closed_index_3d.z + 0*halo + 0,
								   // // // openedDim.x, 
								   // // // openedDim.y, 
								   // // // openedDim.z)];
			// // // if(result !=  deviceSrc[global_index_1d])
			// // // {
				// // // printf("Error in %03d %03d\n",  
					// // // global_index_3d.x, global_index_3d.y, global_index_3d.z);
			// // // }
								
			// // if (closed_index_3d.y < closedDim.y)
			// // {
				// // if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&	
				   // // global_index_3d.y >= 0 && global_index_3d.y < dimy &&
				   // // global_index_3d.x >= 0 && global_index_3d.x < dimx) 
				// // {
					// // //Debug
					// // deviceDst[global_index_1d]= deviceSrc[global_index_1d];
					// // // deviceDst[global_index_1d] = result;
					// // // deviceDst[global_index_1d] =
					// // // sharedMem[at(closed_index_3d.x + 1*halo, 
								 // // // closed_index_3d.y + 1*halo, 
								 // // // closed_index_3d.z + 0*halo,
								 // // // openedDim.x, 
								 // // // openedDim.y, 
								 // // // openedDim.z)];
					
					// // // deviceDst[global_index_1d]= deviceSrc[global_index_1d];
				// // }
			// // }
		// // }
	// // }
	
	
	// // Debug: Naive implementation stencil_3d from global mem
	// float param0, param1, param2, param3;
	// // for(sweep=0; sweep<dimz; sweep++)
	// {
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
									 // // sweep * closedDim.z);
		// ///
		// numThreads = blockDim.x  * blockDim.y  * blockDim.z;
		// openedSize = openedDim.x * openedDim.y * openedDim.z;
		// closedSize = closedDim.x * closedDim.y * closedDim.z;
		
		// ///
		// numReading = (openedSize / numThreads) + ((openedSize % numThreads)?1:0);    
		// numWriting = (closedSize / numThreads) + ((closedSize % numThreads)?1:0);  
		
		
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
							  
							  
			// if(((global_index_3d.z >0) && (global_index_3d.z < (dimz-1))) &&
			   // ((global_index_3d.y >0) && (global_index_3d.y < (dimy-1))) &&
			   // ((global_index_3d.x >0) && (global_index_3d.x < (dimx-1))) )
			// {
				// result = param0 * (deviceSrc[at(global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z+0, dimx, dimy, dimz)]) 
				
					   // + param1 * (deviceSrc[at(global_index_3d.x-1, global_index_3d.y+0, global_index_3d.z+0, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y+0, global_index_3d.z+0, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+0, global_index_3d.y-1, global_index_3d.z+0, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+0, global_index_3d.y+1, global_index_3d.z+0, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z-1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+0, global_index_3d.y+0, global_index_3d.z+1, dimx, dimy, dimz)])
								  
					   // + param2 * (deviceSrc[at(global_index_3d.x-1, global_index_3d.y-1, global_index_3d.z+0, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x-1, global_index_3d.y+1, global_index_3d.z+0, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y-1, global_index_3d.z+0, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y+1, global_index_3d.z+0, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x-1, global_index_3d.y+0, global_index_3d.z-1, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x-1, global_index_3d.y+0, global_index_3d.z+1, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y+0, global_index_3d.z-1, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y+0, global_index_3d.z+1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+0, global_index_3d.y-1, global_index_3d.z-1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+0, global_index_3d.y-1, global_index_3d.z+1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+0, global_index_3d.y+1, global_index_3d.z-1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+0, global_index_3d.y+1, global_index_3d.z+1, dimx, dimy, dimz)])
								  
					   // + param3 * (deviceSrc[at(global_index_3d.x-1, global_index_3d.y-1, global_index_3d.z-1, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x-1, global_index_3d.y-1, global_index_3d.z+1, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x-1, global_index_3d.y+1, global_index_3d.z-1, dimx, dimy, dimz)]
					              // +deviceSrc[at(global_index_3d.x-1, global_index_3d.y+1, global_index_3d.z+1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y-1, global_index_3d.z-1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y-1, global_index_3d.z+1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y+1, global_index_3d.z-1, dimx, dimy, dimz)]
								  // +deviceSrc[at(global_index_3d.x+1, global_index_3d.y+1, global_index_3d.z+1, dimx, dimy, dimz)]);
				// // for(int zz=-1; zz<2; zz++)
				// // {
					// // // sliceSrc = d_src + (index_3d.z + zz) * slicePitch; 
					// // for(int yy=-1; yy<2; yy++)
					// // {
						// // // rowSrc = (float*)(sliceSrc + (index_3d.y + yy) * pitch); 
						// // for(int xx=-1; xx<2; xx++)
						// // {
							// // // if((zz!=0) && (yy!=0) && (xx!=0))
							// // // {
								// // // // tmp += rowSrc[index_3d.x + xx];
								// // // tmp += deviceSrc[at(global_index_3d.x + xx, 
													// // // global_index_3d.x + yy,
													// // // global_index_3d.x + zz,
													// // // dimx, dimy, dimz)];
							// // // }
						// // }
					// // }
				// // }
			// }
			// // sliceSrc = d_src + (index_3d.z) * slicePitch; 
			// // rowSrc = (float*)(sliceSrc + (index_3d.y) * pitch); 
			// // result = alpha*deviceSrc[at(global_index_3d.x + 0, 
										// // global_index_3d.x + 0,
										// // global_index_3d.x + 0,
										// // dimx, dimy, dimz)];
				   // // + beta*tmp;	
								
			// if (closed_index_3d.y < closedDim.y)
			// {
				// if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&	
				   // global_index_3d.y >= 0 && global_index_3d.y < dimy &&
				   // global_index_3d.x >= 0 && global_index_3d.x < dimx) 
				// {
					// //Debug
					// // deviceDst[global_index_1d]= deviceSrc[global_index_1d];
					// deviceDst[global_index_1d] = result;
					// // deviceDst[global_index_1d] =
					// // sharedMem[at(closed_index_3d.x + 1*halo, 
								 // // closed_index_3d.y + 1*halo, 
								 // // closed_index_3d.z + 0*halo,
								 // // openedDim.x, 
								 // // openedDim.y, 
								 // // openedDim.z)];
					
					// // deviceDst[global_index_1d]= deviceSrc[global_index_1d];
				// }
			// }
		// }
	// }// End sweep
	
	
	
	
	
	// // Debug: Write stencil_3d, global mem
	// // for(sweep=0; sweep<dimz; sweep++)
	// // {
	
	// // //Calculate the closed form, instruction parallelism
	// // closedDim  = make_int3(2*blockDim.x,
	 		 		       // // 2*blockDim.y,
						   // // 1*blockDim.z);
	// // openedDim  = make_int3(closedDim.x + 2*halo,
	 					   // // closedDim.y + 2*halo,
						   // // closedDim.z + 0*halo);
						  
	// // offset_index_3d  = make_int3(blockIdx.x * closedDim.x, 
								 // // blockIdx.y * closedDim.y,
								 // // // blockIdx.z * closedDim.z);
								 // // sweep * closedDim.z);
	// // ///
	// // numThreads = blockDim.x  * blockDim.y  * blockDim.z;
	// // openedSize = openedDim.x * openedDim.y * openedDim.z;
	// // closedSize = closedDim.x * closedDim.y * closedDim.z;
	
	// // ///
	// // numReading = (openedSize / numThreads) + ((openedSize % numThreads)?1:0);    
	// // numWriting = (closedSize / numThreads) + ((closedSize % numThreads)?1:0);    
	
	// // #pragma unroll
	// // for(thisReading=0; thisReading<numReading; thisReading++)
	// // {
		// // opened_index_1d =  threadIdx.z * blockDim.y * blockDim.x +                      										
						   // // threadIdx.y * blockDim.x +                                   										
						   // // threadIdx.x +                  
						   // // thisReading * numThreads; //Flatten everything
		// // opened_index_3d = make_int3((opened_index_1d % (openedDim.y*openedDim.x) % openedDim.x),		
								    // // (opened_index_1d % (openedDim.y*openedDim.x) / openedDim.x),		
									// // (opened_index_1d / (openedDim.y*openedDim.x)) );  
		// // global_index_3d = make_int3((offset_index_3d.x + opened_index_3d.x - 1*halo),
									// // (offset_index_3d.y + opened_index_3d.y - 1*halo),
									// // (offset_index_3d.z + opened_index_3d.z - 0*halo) );
		// // global_index_1d = global_index_3d.z * dimy * dimx +
						  // // global_index_3d.y * dimx +
						  // // global_index_3d.x;
		// // if (opened_index_3d.y < openedDim.y)
		// // {
			// // if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&	
			   // // global_index_3d.y >= 0 && global_index_3d.y < dimy &&
			   // // global_index_3d.x >= 0 && global_index_3d.x < dimx) 
			// // {
				// // sharedMem[at(opened_index_3d.x, 
							 // // opened_index_3d.y, 
							 // // opened_index_3d.z,
						     // // openedDim.x, 
							 // // openedDim.y, 
							 // // openedDim.z)]
				// // = deviceSrc[global_index_1d];
				
				// // //Debug
				// // // deviceDst[global_index_1d]= deviceSrc[global_index_1d];
			// // }
		// // }
	// // }
	// // __syncthreads();
	
	
	// // #pragma unroll
	// // for(thisWriting=0; thisWriting<numWriting; thisWriting++)
	// // {
		// // closed_index_1d =  threadIdx.z * blockDim.y * blockDim.x +                      										
						   // // threadIdx.y * blockDim.x +                                   										
						   // // threadIdx.x +                  
						   // // thisWriting * numThreads; //Magic is here 
		// // closed_index_3d = make_int3((closed_index_1d % (closedDim.y*closedDim.x) % closedDim.x),		
								    // // (closed_index_1d % (closedDim.y*closedDim.x) / closedDim.x),		
									// // (closed_index_1d / (closedDim.y*closedDim.x)) );  
		// // global_index_3d = make_int3((offset_index_3d.x + closed_index_3d.x),
									// // (offset_index_3d.y + closed_index_3d.y),
									// // (offset_index_3d.z + closed_index_3d.z) );
		// // global_index_1d = global_index_3d.z * dimy * dimx +
						  // // global_index_3d.y * dimx +
						  // // global_index_3d.x;
						  
						  
		// // result	= sharedMem[at(closed_index_3d.x + 1*halo + 0, 
							   // // closed_index_3d.y + 1*halo + 0, 
							   // // closed_index_3d.z + 0*halo + 0,
						       // // openedDim.x, 
							   // // openedDim.y, 
							   // // openedDim.z)];
		// // // if(result !=  deviceSrc[global_index_1d])
		// // // {
			// // // printf("Error in %03d %03d\n",  
				// // // global_index_3d.x, global_index_3d.y, global_index_3d.z);
		// // // }
							
		// // if (closed_index_3d.y < closedDim.y)
		// // {
			// // if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&	
			   // // global_index_3d.y >= 0 && global_index_3d.y < dimy &&
			   // // global_index_3d.x >= 0 && global_index_3d.x < dimx) 
			// // {
				// // //Debug
				// // // deviceDst[global_index_1d]= deviceSrc[global_index_1d];
				// // deviceDst[global_index_1d] = result;
				// // // deviceDst[global_index_1d] =
				// // // sharedMem[at(closed_index_3d.x + 1*halo, 
							 // // // closed_index_3d.y + 1*halo, 
							 // // // closed_index_3d.z + 0*halo,
						     // // // openedDim.x, 
							 // // // openedDim.y, 
							 // // // openedDim.z)];
				
				// // // deviceDst[global_index_1d]= deviceSrc[global_index_1d];
			// // }
		// // }
	// // }
	
	// // }
// }                                                                                         
