#include "add.hpp"
#include <cuComplex.h>
#include "utility.hpp"
#include "helper_math.h"
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace csmri
{
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef blockDimx
#define blockDimx 16		
#endif

#ifndef blockDimy
#define blockDimy 16		
#endif

#ifndef blockDimz
#define blockDimz 1		
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void __add(
	float2* inA,	
	float2* inB,	
	float2* out,	
	int dimx,
	int dimy,
	int dimz)
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
	
	//Do computing
	out[index] = cuCaddf(inA[index], inB[index]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void add(
	float2* inA,	
	float2* inB,	
	float2* out,	
	int dimx,
	int dimy,
	int dimz)
{
	dim3 numBlocks(
		(dimx/blockDimx + ((dimx%blockDimx)?1:0)),
		(dimy/blockDimy + ((dimy%blockDimy)?1:0)),
		(dimz/blockDimz + ((dimz%blockDimz)?1:0)) );
	dim3 numThreads(blockDimx, blockDimy, blockDimz);
	__add<<<numBlocks, numThreads>>>(inA, inB, out, dimx, dimy, dimz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void __add(
	float2* inA,	
	float2* inB,	
	float2* inC,
	float2* inD,
	float2* inE,
	float2* out,	
	int dimx,
	int dimy,
	int dimz)
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
	
	//Do computing
	out[index] = inA[index] + inB[index] + inC[index] + inD[index] + inE[index];
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void add(
	float2* inA,	
	float2* inB,	
	float2* inC,
	float2* inD,
	float2* inE,
	float2* out,	
	int dimx,
	int dimy,
	int dimz,
	cudaStream_t stream)
{
	dim3 numBlocks(
		(dimx/blockDimx + ((dimx%blockDimx)?1:0)),
		(dimy/blockDimy + ((dimy%blockDimy)?1:0)),
		(dimz/blockDimz + ((dimz%blockDimz)?1:0)) );
	dim3 numThreads(blockDimx, blockDimy, blockDimz);
	// __add<<<numBlocks, numThreads>>>(inA, inB, inC, inD, inE, out, dimx, dimy, dimz);
	__add<<<numBlocks, numThreads, 0, stream>>>(inA, inB, inC, inD, inE, out, dimx, dimy, dimz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
}