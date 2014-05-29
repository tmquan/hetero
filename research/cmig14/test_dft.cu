#include <iostream>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "dft.hpp"
#include "timer.hpp"
#include "utility.hpp"
#include "helper_math.h"
#include <hetero_cmdparser.hpp>
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace csmri;
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void	warmUp()	{}
////////////////////////////////////////////////////////////////////////////////////////////////////	
const char* key =
	"{ h      |help        |      | print help message }"	
	"{ dimx   |dimx        |      | dimensionx }"
	"{ dimy   |dimy        |      | dimensiony }"
	"{ dimz   |dimz        |      | dimensionz }"
	"{ srcFDFT|srcFDFT     |      | source File }"
	"{ dstFDFT|dstFDFT     |      | destination File }"
	"{ dstIDFT|dstIDFT     |      | destination File }"
	;
////////////////////////////////////////////////////////////////////////////////////////////////////	
int main(int argc, char** argv)
{
	cudaSetDevice(0);
	cudaDeviceReset();
	warmUp<<<1, 1>>>();
	// Retrieve the number of execId
	// Parsing the arguments
	CommandLineParser cmd(argc, argv, key);
	const int dimx  			= cmd.get<int>("dimx", false);
	const int dimy  			= cmd.get<int>("dimy", false);
	const int dimz  			= cmd.get<int>("dimz", false);
	/// Calculate the total size
	const int total = dimx*dimy*dimz;
	
	const string srcFDFT		= cmd.get<string>("srcFDFT", false);
	const string dstFDFT		= cmd.get<string>("dstFDFT", false);
	const string dstIDFT		= cmd.get<string>("dstIDFT", false);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	GpuTimer timer;
	float2 *h_src, *h_dst;

	h_src = new float2[total];
	h_dst = new float2[total];
	
	//Open the file
	// checkReadFile("lenna_128x128x256_full.bin", h_src, total*sizeof(float2));
	checkReadFile(srcFDFT, h_src, total*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float2 *d_src;
	cudaMalloc((void**)&d_src, total*sizeof(float2));
	
	float2 *d_dst;
	cudaMalloc((void**)&d_dst, total*sizeof(float2));
	
	cudaMemcpy(d_src, h_src, total*sizeof(float2), cudaMemcpyHostToDevice);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Create Fourier plan 2.5d. </summary>
	cufftHandle plan;
	
	int rank[3] = {dimx, dimy, dimz};
	cufftPlanMany(&plan,
		2,			//Dimensionality of the transform (1, 2, or 3)
		rank,		//Array of size rank, describing the size of each dimension
		NULL,
		1,			//Distance between two successive input elements in the innermost dimension
		dimy*dimx,  //Distance between the first element of two consecutive signals in a batch of the input data
		NULL,
		1,
		dimy*dimx,
		CUFFT_C2C,
		dimz);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Along x
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Forward Fourier. </summary>
	timer.Start();
	/// TODO: Call the kernel here
	dft(d_src, d_dst, dimx, dimy, dimz, DFT_FORWARD, plan);
	timer.Stop();
	printf("Forward Fourier Transform: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	// checkWriteFile("lenna_128x128x256_fdft.bin", h_dst, total*sizeof(float2));
	checkWriteFile(dstFDFT, h_dst, total*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMemcpy(d_src, d_dst, total*sizeof(float2), cudaMemcpyDeviceToDevice);
	/// <summary>	Inverse Fourier. </summary>
	timer.Start();
	dft(d_src, d_dst, dimx, dimy, dimz, DFT_INVERSE, plan);
	timer.Stop();
	printf("Inverse Fourier Transform: %4.4f ms\n", timer.Elapsed());
	
	timer.Start();
	scale(d_dst, d_dst, dimx, dimy, dimz, 1.0f/(dimx*dimy) );
	timer.Stop();
	printf("Scaling: %4.4f ms\n", timer.Elapsed());
	
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	// checkWriteFile("lenna_128x128x256_idft.bin", h_dst, total*sizeof(float2));
	checkWriteFile(dstIDFT, h_dst, total*sizeof(float2));
	return 0;
}