#include "div.hpp"
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
void __div(
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

	out[index] = cuCdivf(inA[index], inB[index]);
	// //Do computing
	// if(cuCabsf(inB[index]) != 0.0f)
		// out[index] = cuCdivf(inA[index], inB[index]);
	// else
		// out[index] = inA[index];
		
	// // Do computing
	// if(cuCabsf(inB[index]) != 0.0f)
		// out[index] = make_float2(inA[index].x/inB[index].x, inA[index].y/inB[index].y);
	// else
		// out[index] = inA[index];
		
		
	// cuFloatComplex x = inA[index];
	// cuFloatComplex y = inB[index];
	// cuFloatComplex quot;
    // float s = ((float)fabs((double)cuCrealf(y))) + 
              // ((float)fabs((double)cuCimagf(y)));
    // float oos = 1.0f / s;
    // float ars = cuCrealf(x) * oos;
    // float ais = cuCimagf(x) * oos;
    // float brs = cuCrealf(y) * oos;
    // float bis = cuCimagf(y) * oos;
    // s = (brs * brs) + (bis * bis);
    // oos = 1.0f / s;
    // quot = make_cuFloatComplex (((ars * brs) + (ais * bis)) * oos,
                                // ((ais * brs) - (ars * bis)) * oos);
    // // return quot;
	// out[index] = quot;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void div(
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
	__div<<<numBlocks, numThreads>>>(inA, inB, out, dimx, dimy, dimz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
}