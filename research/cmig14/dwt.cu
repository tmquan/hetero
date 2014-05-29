#include "dwt.hpp"
#include "utility.hpp"
#include "helper_math.h"
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace csmri
{
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef blockDimx
#define blockDimx 8		
#endif

#ifndef blockDimy
#define blockDimy 8		
#endif

#ifndef blockDimz
#define blockDimz 2		
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
// If 1 -> 0x 80000000 80000000: negate 2 numbers inside float2
__host__ __device__ static __inline__  
void switchSign(unsigned int intSign, float2* number)
{
	*((long long int*)(number)) 
		^= intSign ? 0x8000000080000000:0x0000000000000000;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ 
void __encode_8(
	float2* src, 
	float2* dst, 
	int dimx, int dimy, int dimz)
{
	//3D global index
	int3 idx = make_int3(
		blockIdx.x*blockDim.x+threadIdx.x,
		blockIdx.y*blockDim.y+threadIdx.y,
		blockIdx.z*blockDim.z+threadIdx.z);
	
	//1D global index
	int index 	= 	idx.z*dimy*dimx		
				+	idx.y*dimx				
				+	idx.x;				
									  
	//Check valid indices
	if (idx.x >= dimx || idx.y >= dimy || idx.z >= dimz)
		return;
	
	float2 a, b, c, d;
	__shared__ float2 sMem[blockDimz][blockDimy][blockDimx];
	
	sMem[threadIdx.z][threadIdx.y][threadIdx.x] = src[index];
	__syncthreads();
	/**********************************************************************************************/
	if(((threadIdx.y&0)==0)&&((threadIdx.x&0)==0))
	{	
		a = sMem[threadIdx.z][(threadIdx.y & (~1)) + 0][(threadIdx.x & (~1)) + 0];
		b = sMem[threadIdx.z][(threadIdx.y & (~1)) + 0][(threadIdx.x & (~1)) + 1];
		c = sMem[threadIdx.z][(threadIdx.y & (~1)) + 1][(threadIdx.x & (~1)) + 0];
		d = sMem[threadIdx.z][(threadIdx.y & (~1)) + 1][(threadIdx.x & (~1)) + 1];
		//__threadfence_block();
		switchSign((((threadIdx.y>>0 & 1) & 0) ^ ((threadIdx.x>>0 & 1) & 0)), &a);
		switchSign((((threadIdx.y>>0 & 1) & 0) ^ ((threadIdx.x>>0 & 1) & 1)), &b);
		switchSign((((threadIdx.y>>0 & 1) & 1) ^ ((threadIdx.x>>0 & 1) & 0)), &c);
		switchSign((((threadIdx.y>>0 & 1) & 1) ^ ((threadIdx.x>>0 & 1) & 1)), &d);
		//__threadfence_block();
		sMem[threadIdx.z][threadIdx.y][threadIdx.x] = (0.5f)*(a + b + c + d);
	}
	__syncthreads();
	/**********************************************************************************************/
	if(((threadIdx.y&1)==0)&&((threadIdx.x&1)==0))
	{
		a = sMem[threadIdx.z][(threadIdx.y & (~3)) + 0][(threadIdx.x & (~3)) + 0];
		b = sMem[threadIdx.z][(threadIdx.y & (~3)) + 0][(threadIdx.x & (~3)) + 2];
		c = sMem[threadIdx.z][(threadIdx.y & (~3)) + 2][(threadIdx.x & (~3)) + 0];
		d = sMem[threadIdx.z][(threadIdx.y & (~3)) + 2][(threadIdx.x & (~3)) + 2];
		//__threadfence_block();
		switchSign((((threadIdx.y>>1 & 1) & 0) ^ ((threadIdx.x>>1 & 1) & 0)), &a);
		switchSign((((threadIdx.y>>1 & 1) & 0) ^ ((threadIdx.x>>1 & 1) & 1)), &b);
		switchSign((((threadIdx.y>>1 & 1) & 1) ^ ((threadIdx.x>>1 & 1) & 0)), &c);
		switchSign((((threadIdx.y>>1 & 1) & 1) ^ ((threadIdx.x>>1 & 1) & 1)), &d);
		//__threadfence_block();
		sMem[threadIdx.z][threadIdx.y][threadIdx.x] = (0.5f)*(a + b + c + d);
	}
 	__syncthreads();
	/**********************************************************************************************/
	if(((threadIdx.y&3)==0)&&((threadIdx.x&3)==0))
	{
		a = sMem[threadIdx.z][(threadIdx.y & (~7)) + 0][(threadIdx.x & (~7)) + 0];
		b = sMem[threadIdx.z][(threadIdx.y & (~7)) + 0][(threadIdx.x & (~7)) + 4];
		c = sMem[threadIdx.z][(threadIdx.y & (~7)) + 4][(threadIdx.x & (~7)) + 0];
		d = sMem[threadIdx.z][(threadIdx.y & (~7)) + 4][(threadIdx.x & (~7)) + 4];
		__threadfence();
		switchSign((((threadIdx.y>>2 & 1) & 0) ^ ((threadIdx.x>>2 & 1) & 0)), &a);
		switchSign((((threadIdx.y>>2 & 1) & 0) ^ ((threadIdx.x>>2 & 1) & 1)), &b);
		switchSign((((threadIdx.y>>2 & 1) & 1) ^ ((threadIdx.x>>2 & 1) & 0)), &c);
		switchSign((((threadIdx.y>>2 & 1) & 1) ^ ((threadIdx.x>>2 & 1) & 1)), &d);
		__threadfence();
		sMem[threadIdx.z][threadIdx.y][threadIdx.x] = (0.5f)*(a + b + c + d);
	}
	__syncthreads();
	/**********************************************************************************************/
	dst[index] = sMem[threadIdx.z][threadIdx.y][threadIdx.x];
}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ 
void __decode_8(
	float2* src, 
	float2* dst, 
	int dimx, int dimy, int dimz)
{
	//3D global index
	int3 idx = make_int3(
		blockIdx.x*blockDim.x+threadIdx.x,
		blockIdx.y*blockDim.y+threadIdx.y,
		blockIdx.z*blockDim.z+threadIdx.z);
	
	//1D global index
	int index 	= 	idx.z*dimy*dimx		
				+	idx.y*dimx				
				+	idx.x;				
									  
	//Check valid indices
	if (idx.x >= dimx || idx.y >= dimy || idx.z >= dimz)
		return;
	
	float2 a, b, c, d;
	__shared__ float2 sMem[blockDimz][blockDimy][blockDimx];

	sMem[threadIdx.z][threadIdx.y][threadIdx.x] = src[index];
	__syncthreads();
	/**********************************************************************************************/
	if(((threadIdx.y&3)==0)&&((threadIdx.x&3)==0))
	{
		a = sMem[threadIdx.z][(threadIdx.y & (~7)) + 0][(threadIdx.x & (~7)) + 0];
		b = sMem[threadIdx.z][(threadIdx.y & (~7)) + 0][(threadIdx.x & (~7)) + 4];
		c = sMem[threadIdx.z][(threadIdx.y & (~7)) + 4][(threadIdx.x & (~7)) + 0];
		d = sMem[threadIdx.z][(threadIdx.y & (~7)) + 4][(threadIdx.x & (~7)) + 4];
		__threadfence();
		switchSign((((threadIdx.y>>2 & 1) & 0) ^ ((threadIdx.x>>2 & 1) & 0)), &a);
		switchSign((((threadIdx.y>>2 & 1) & 0) ^ ((threadIdx.x>>2 & 1) & 1)), &b);
		switchSign((((threadIdx.y>>2 & 1) & 1) ^ ((threadIdx.x>>2 & 1) & 0)), &c);
		switchSign((((threadIdx.y>>2 & 1) & 1) ^ ((threadIdx.x>>2 & 1) & 1)), &d);
		__threadfence();
		sMem[threadIdx.z][threadIdx.y][threadIdx.x] = (0.5f)*(a + b + c + d);
	}
	__syncthreads();
	/**********************************************************************************************/
	if(((threadIdx.y&1)==0)&&((threadIdx.x&1)==0))
	{
		a = sMem[threadIdx.z][(threadIdx.y & (~3)) + 0][(threadIdx.x & (~3)) + 0];
		b = sMem[threadIdx.z][(threadIdx.y & (~3)) + 0][(threadIdx.x & (~3)) + 2];
		c = sMem[threadIdx.z][(threadIdx.y & (~3)) + 2][(threadIdx.x & (~3)) + 0];
		d = sMem[threadIdx.z][(threadIdx.y & (~3)) + 2][(threadIdx.x & (~3)) + 2];
		//__threadfence_block();
		switchSign((((threadIdx.y>>1 & 1) & 0) ^ ((threadIdx.x>>1 & 1) & 0)), &a);
		switchSign((((threadIdx.y>>1 & 1) & 0) ^ ((threadIdx.x>>1 & 1) & 1)), &b);
		switchSign((((threadIdx.y>>1 & 1) & 1) ^ ((threadIdx.x>>1 & 1) & 0)), &c);
		switchSign((((threadIdx.y>>1 & 1) & 1) ^ ((threadIdx.x>>1 & 1) & 1)), &d);
		//__threadfence_block();
		sMem[threadIdx.z][threadIdx.y][threadIdx.x] = (0.5f)*(a + b + c + d);
	}
 	__syncthreads();
	/**********************************************************************************************/
	if(((threadIdx.y&0)==0)&&((threadIdx.x&0)==0))
	{	
		a = sMem[threadIdx.z][(threadIdx.y & (~1)) + 0][(threadIdx.x & (~1)) + 0];
		b = sMem[threadIdx.z][(threadIdx.y & (~1)) + 0][(threadIdx.x & (~1)) + 1];
		c = sMem[threadIdx.z][(threadIdx.y & (~1)) + 1][(threadIdx.x & (~1)) + 0];
		d = sMem[threadIdx.z][(threadIdx.y & (~1)) + 1][(threadIdx.x & (~1)) + 1];
		//__threadfence_block();
		switchSign((((threadIdx.y>>0 & 1) & 0) ^ ((threadIdx.x>>0 & 1) & 0)), &a);
		switchSign((((threadIdx.y>>0 & 1) & 0) ^ ((threadIdx.x>>0 & 1) & 1)), &b);
		switchSign((((threadIdx.y>>0 & 1) & 1) ^ ((threadIdx.x>>0 & 1) & 0)), &c);
		switchSign((((threadIdx.y>>0 & 1) & 1) ^ ((threadIdx.x>>0 & 1) & 1)), &d);
		//__threadfence_block();
		sMem[threadIdx.z][threadIdx.y][threadIdx.x] = (0.5f)*(a + b + c + d);
	}
	__syncthreads();
	/**********************************************************************************************/
	dst[index] = sMem[threadIdx.z][threadIdx.y][threadIdx.x];
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void encode(
	float2* src,	
	float2* dst,	
	int dimx,
	int dimy,
	int dimz)
{
	dim3 numBlocks(
		(dimx/blockDimx + ((dimx%blockDimx)?1:0)),
		(dimy/blockDimy + ((dimy%blockDimy)?1:0)),
		(dimz/blockDimz + ((dimz%blockDimz)?1:0)) );
	dim3 numThreads(blockDimx, blockDimy, blockDimz);
	__encode_8<<<numBlocks, numThreads>>>(src, dst, dimx, dimy, dimz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void decode(
	float2* src,	
	float2* dst,	
	int dimx,
	int dimy,
	int dimz)
{
	dim3 numBlocks(
		(dimx/blockDimx + ((dimx%blockDimx)?1:0)),
		(dimy/blockDimy + ((dimy%blockDimy)?1:0)),
		(dimz/blockDimz + ((dimz%blockDimz)?1:0)) );
	dim3 numThreads(blockDimx, blockDimy, blockDimz);
	__decode_8<<<numBlocks, numThreads>>>(src, dst, dimx, dimy, dimz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void dwt(
	float2* src,	
	float2* dst,	
	int dimx,
	int dimy,
	int dimz,
	int flag)
{
	/// <summary>	Perform Wavelet Transform </summary>
	switch(flag)
	{
	/// <summary>	Forward Wavelet Transform </summary>
	case DWT_FORWARD:		
		encode(src, dst, dimx, dimy, dimz);
		//checkCudaErrors(cudaGetLastError());
		break;
	
	/// <summary>	Inverse Wavelet Transform </summary>
	case DWT_INVERSE:		
		decode(src, dst, dimx, dimy, dimz);
		//checkCudaErrors(cudaGetLastError());
		break;
	default:
		break;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
}