#include <iostream>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////////////////////////
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "sum.hpp"
#include "timer.hpp"
#include "utility.hpp"
#include "helper_math.h"
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace csmri;
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void	warmUp()	{}
////////////////////////////////////////////////////////////////////////////////////////////////////	
int main(int argc, char** argv)
{
	cudaSetDevice(0);
	cudaDeviceReset();
	warmUp<<<1, 1>>>();

	int dimx = 128;
	int dimy = 128;
	int dimz = 256;
	/// Calculate the total size
	int total = dimx*dimy*dimz;
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	GpuTimer timer;
	float2 *h_A, *h_B, *h_C, *h_D;
	float2 *d_A, *d_B;
	
	fstream hSumFile, dSumFile;
	h_A = new float2[total];
	h_B = new float2[1];
	h_C = new float2[1];
	
	
	//Generate data for testing
	srand(time(NULL));
	for(int i=0; i<total; i++)
	{
		h_A[i] = make_float2( (float)rand()/RAND_MAX, (float)rand()/RAND_MAX );
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	//! We use Kahan summation for an accurate sum of large arrays.
	//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
	// CPU summation
	h_B[0] = h_A[0];
	float2 c = make_float2( 0.0f, 0.0f );
	
	// Start at index 1
	for(int i=1; i<total; i++)
	{
		float2 y = h_A[i] - c;
		float2 t = h_B[0] + y;
		c = (t - h_B[0]) - y;
        h_B[0] = t;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	//GPU summation
	cudaMalloc((void**)&d_A, total*sizeof(float2));
	cudaMalloc((void**)&d_B, total*sizeof(float2));
	
	cudaMemcpy(d_A, h_A, total*sizeof(float2), cudaMemcpyHostToDevice);
	// cudaMemset(d_B, 0, total*sizeof(float2));
	
	float2 init = make_float2(0,0);
	timer.Start();
	float2 result = thrust::reduce(thrust::device_pointer_cast(d_A), 
		thrust::device_pointer_cast(d_A+total), 
		init);
	// sum(d_A, d_B, dimx, dimy, dimz);
	timer.Stop();
	// cudaMemcpy(h_C, d_B, 1*sizeof(float2), cudaMemcpyDeviceToHost);
	printf("Sumation reduction: %4.4f ms\n", timer.Elapsed());	
	printf("CPU sum: %4.4f\t%4.4f\n", h_B[0].x, h_B[0].y);
	// printf("GPU sum: %4.4f\t%4.4f\n", h_C[0].x, h_C[0].y);
	printf("GPU sum: %4.4f\t%4.4f\n", result.x, result.y);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	return 0;
}