#include <stdio.h>
#include "dft.hpp"
#include "utility.hpp"
#include "helper_math.h"
#include <cufft.h>

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
void __scale(	
	float2 *src, 
	float2 *dst,
	int dimx,
	int dimy,
	int dimz, 
	float factor)
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

	//Do scaling
	dst[index] = factor*src[index];
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void scale(
	float2 *src, 
	float2 *dst,
	int dimx,
	int dimy,
	int dimz, 
	float factor)
{
	dim3 numBlocks(
		(dimx/blockDimx + ((dimx%blockDimx)?1:0)),
		(dimy/blockDimy + ((dimy%blockDimy)?1:0)),
		(dimz/blockDimz + ((dimz%blockDimz)?1:0)) );
	dim3 numThreads(blockDimx, blockDimy, blockDimz);
	__scale<<<numBlocks, numThreads>>>(src, dst, dimx, dimy, dimz, factor);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void dft(
	float2* src,	
	float2* dst,	
	int dimx,
	int dimy,
	int dimz,
	int flag,
	cufftHandle plan)
{
	switch(flag)
	{
	/// <summary>	Forward Fourier Transform </summary>
	case DFT_FORWARD:
		checkCufftError(
		cufftExecC2C(plan, 
			src,
			dst,
			CUFFT_FORWARD) );
		// checkLastErrors();
		break;
	
	/// <summary>	Inverse Fourier Transform </summary>
	case DFT_INVERSE:		
		checkCufftError(
		cufftExecC2C(plan, 
			src,
			dst,
			CUFFT_INVERSE) );
		// checkLastErrors();
		/// <summary>	Scale the output			 </summary>
		// 	scale25d(dst, nRows, nCols, nTems, 1.0f/(nRows*nCols));
		// 	checkLastErrors();
		/// <comment>	Need to be scaled explicitly </comment>
		break;
	default:
		break;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
}