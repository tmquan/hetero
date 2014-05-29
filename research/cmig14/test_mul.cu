#include <iostream>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "mul.hpp"
#include "timer.hpp"
#include "utility.hpp"
#include "helper_math.h"
#include "cuComplex.h" // for complex multiplication
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
	int total = dimy*dimx*dimz;
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	GpuTimer timer;
	float2 *h_A, *h_B, *h_C, *h_D;
	float2 *d_A, *d_B, *d_C;
	
	fstream hAddFile, dAddFile;
	h_A = new float2[total];
	h_B = new float2[total];
	h_C = new float2[total];
	h_D = new float2[total];
	//Generate data for testing
	srand(time(NULL));
	for(int i=0; i<total; i++)
	{
		h_A[i] = make_float2( (float)rand()/RAND_MAX, (float)rand()/RAND_MAX );
		h_B[i] = make_float2( (float)rand()/RAND_MAX, (float)rand()/RAND_MAX );
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////
	//CPU Complex Multiplication
	for(int i=0; i<total; i++)
	{
		h_C[i] = cuCmulf(h_A[i], h_B[i]);
		// h_C[i] 	= make_float2(h_A[i].x * h_B[i].x - h_A[i].y * h_B[i].y,
							  // h_A[i].x * h_B[i].y + h_A[i].y * h_B[i].x);
	}

	checkWriteFile("hMul.bin", h_C, total*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	//GPU Complex Multiplication
	cudaMalloc((void**)&d_A, total*sizeof(float2));
	cudaMalloc((void**)&d_B, total*sizeof(float2));
	cudaMalloc((void**)&d_C, total*sizeof(float2));
	cudaMemcpy(d_A, h_A, total*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, total*sizeof(float2), cudaMemcpyHostToDevice);
	
	timer.Start();
	mul(d_A, d_B, d_C, dimx, dimy, dimz);
	timer.Stop();
	
	printf("Complex Multiplication: %4.4f ms\n", timer.Elapsed());
	
	cudaMemcpy(h_D, d_C, total*sizeof(float2), cudaMemcpyDeviceToHost);
	// Host resuls and Device results are difference in term of precision
	checkWriteFile("dMul.bin", h_D, total*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	return 0;
}