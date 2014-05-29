#include "ddt.hpp"
#include "utility.hpp"
#include "helper_math.h"
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
/// Mirror effect, acts like Neumann Boundary Condition
#define at(x, y, z, dimx, dimy, dimz) (clamp(z, 0, dimz-1)*dimy*dimx		\
									  +clamp(y, 0, dimy-1)*dimx				\
									  +clamp(x, 0, dimx-1))				
////////////////////////////////////////////////////////////////////////////////////////////////////
/// Do not need to use shared memory because computation is small, and reading is dominated
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void __dxt_forward(
	float2* u, 
	float2* dx,
	int dimx, 
	int dimy, 
	int dimz)
{
	//3D global index
	int3 idx = make_int3(
		blockIdx.x*blockDim.x+threadIdx.x,
		blockIdx.y*blockDim.y+threadIdx.y,
		blockIdx.z*blockDim.z+threadIdx.z);
	
	//Check valid indices
	if (idx.x >= dimx || idx.y >= dimy || idx.z >= dimz)
		return;
	
	//
	dx[at(idx.x, idx.y, idx.z, dimx, dimy, dimz)]
	=  0.5f	*	(u[at(idx.x+1, idx.y, idx.z, dimx, dimy, dimz)]
			    -u[at(idx.x-1, idx.y, idx.z, dimx, dimy, dimz)]);	
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void dxt_forward(
	float2* u, 
	float2* dx,
	int dimx, 
	int dimy, 
	int dimz)
{
	dim3 numBlocks((dimx/blockDimx + ((dimx%blockDimx)?1:0)),
				   (dimy/blockDimy + ((dimy%blockDimy)?1:0)),
				   (dimz/blockDimz + ((dimz%blockDimz)?1:0)) );
	dim3 numThreads(blockDimx, blockDimy, blockDimz);
	__dxt_forward<<<numBlocks, numThreads>>>(u, dx, dimx, dimy, dimz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void __dxt_inverse(
	float2* u, 
	float2* dx,
	int dimx, 
	int dimy, 
	int dimz)
{
	//3D global index
	int3 idx = make_int3(
		blockIdx.x*blockDim.x+threadIdx.x,
		blockIdx.y*blockDim.y+threadIdx.y,
		blockIdx.z*blockDim.z+threadIdx.z);
	
	//Check valid indices
	if (idx.x >= dimx || idx.y >= dimy || idx.z >= dimz)
		return;
	
	//
	dx[at(idx.x, idx.y, idx.z, dimx, dimy, dimz)]
	=  0.5f	*	(u[at(idx.x-1, idx.y, idx.z, dimx, dimy, dimz)]
			    -u[at(idx.x+1, idx.y, idx.z, dimx, dimy, dimz)]);	
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void dxt_inverse(
	float2* u, 
	float2* dx,
	int dimx, 
	int dimy, 
	int dimz)
{
	dim3 numBlocks((dimx/blockDimx + ((dimx%blockDimx)?1:0)),
				   (dimy/blockDimy + ((dimy%blockDimy)?1:0)),
				   (dimz/blockDimz + ((dimz%blockDimz)?1:0)) );
	dim3 numThreads(blockDimx, blockDimy, blockDimz);
	__dxt_inverse<<<numBlocks, numThreads>>>(u, dx, dimx, dimy, dimz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void __dxt_laplacian(
	float2* u, 
	float2* dx,
	int dimx, 
	int dimy, 
	int dimz)
{
	//3D global index
	int3 idx = make_int3(
		blockIdx.x*blockDim.x+threadIdx.x,
		blockIdx.y*blockDim.y+threadIdx.y,
		blockIdx.z*blockDim.z+threadIdx.z);
	
	//Check valid indices
	if (idx.x >= dimx || idx.y >= dimy || idx.z >= dimz)
		return;
	
	//
	dx[at(idx.x, idx.y, idx.z, dimx, dimy, dimz)]
	=  2.0f*u[at(idx.x,   idx.y, idx.z, dimx, dimy, dimz)]
		   -u[at(idx.x-1, idx.y, idx.z, dimx, dimy, dimz)]
		   -u[at(idx.x+1, idx.y, idx.z, dimx, dimy, dimz)];	
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void dxt_laplacian(
	float2* u, 
	float2* dx,
	int dimx, 
	int dimy, 
	int dimz)
{
	dim3 numBlocks((dimx/blockDimx + ((dimx%blockDimx)?1:0)),
				   (dimy/blockDimy + ((dimy%blockDimy)?1:0)),
				   (dimz/blockDimz + ((dimz%blockDimz)?1:0)) );
	dim3 numThreads(blockDimx, blockDimy, blockDimz);
	__dxt_laplacian<<<numBlocks, numThreads>>>(u, dx, dimx, dimy, dimz);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void dxt(
	float2* u, 
	float2* dx,
	int dimx, 
	int dimy, 
	int dimz,
	int flag)
{
	switch(flag)
	{
	case DDT_FORWARD:		
		dxt_forward(u, dx, dimx, dimy, dimz);
		break;
	case DDT_INVERSE:		
		dxt_inverse(u, dx, dimx, dimy, dimz);
		break;
	case DDT_LAPLACIAN:		
		dxt_laplacian(u, dx, dimx, dimy, dimz);
		break;
	default:
		break;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
}