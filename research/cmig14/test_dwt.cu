#include <iostream>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "dwt.hpp"
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
	"{ srcFDWT|srcFDWT     |      | source File }"
	"{ dstFDWT|dstFDWT     |      | destination File }"
	"{ dstIDWT|dstIDWT     |      | destination File }"
	;
////////////////////////////////////////////////////////////////////////////////////////////////////	
int main(int argc, char** argv)
{
	cudaSetDevice(0);
	cudaDeviceReset();
	warmUp<<<1, 1>>>();
	CommandLineParser cmd(argc, argv, key);
	const int dimx  			= cmd.get<int>("dimx", false);
	const int dimy  			= cmd.get<int>("dimy", false);
	const int dimz  			= cmd.get<int>("dimz", false);
	/// Calculate the total size
	const int total = dimx*dimy*dimz;
	
	const string srcFDWT		= cmd.get<string>("srcFDWT", false);
	const string dstFDWT		= cmd.get<string>("dstFDWT", false);
	const string dstIDWT		= cmd.get<string>("dstIDWT", false);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	GpuTimer timer;
	float2 *h_src, *h_dst;

	h_src = new float2[total];
	h_dst = new float2[total];
	
	//Open the file
	// checkReadFile("lenna_128x128x256_full.bin", h_src, total*sizeof(float2));
	checkReadFile(srcFDWT, h_src, total*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float2 *d_src;
	cudaMalloc((void**)&d_src, total*sizeof(float2));
	
	float2 *d_dst;
	cudaMalloc((void**)&d_dst, total*sizeof(float2));
	
	cudaMemcpy(d_src, h_src, total*sizeof(float2), cudaMemcpyHostToDevice);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Forward Wavelet. </summary>
	timer.Start();
	/// TODO: Call the kernel here
	dwt(d_src, d_dst, dimx, dimy, dimz, DWT_FORWARD);
	// dwt(d_src, d_src, dimx, dimy, dimz, DWT_FORWARD);
	timer.Stop();
	printf("Forward Wavelet Transform: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_dst, d_src, total*sizeof(float2), cudaMemcpyDeviceToHost);
	// checkWriteFile("lenna_128x128x256_fdwt.bin", h_dst, total*sizeof(float2));
	checkWriteFile(dstFDWT, h_dst, total*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMemcpy(d_src, d_dst, total*sizeof(float2), cudaMemcpyDeviceToDevice);
	/// <summary>	Inverse Wavelet. </summary>
	timer.Start();
	dwt(d_src, d_dst, dimx, dimy, dimz, DWT_INVERSE);
	timer.Stop();
	printf("Inverse Wavelet Transform: %4.4f ms\n", timer.Elapsed());

	
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	// checkWriteFile("lenna_128x128x256_idwt.bin", h_dst, total*sizeof(float2));
	checkWriteFile(dstIDWT, h_dst, total*sizeof(float2));
	return 0;
}