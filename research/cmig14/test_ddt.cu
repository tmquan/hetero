#include <iostream>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "ddt.hpp"
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
	float2 *h_src, *h_dst;

	h_src = new float2[total];
	h_dst = new float2[total];
	
	//Open the file
	checkReadFile("lenna_128x128x256_full.bin", h_src, total*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float2 *d_src;
	cudaMalloc((void**)&d_src, total*sizeof(float2));
	
	float2 *d_dst;
	cudaMalloc((void**)&d_dst, total*sizeof(float2));
	
	cudaMemcpy(d_src, h_src, total*sizeof(float2), cudaMemcpyHostToDevice);
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Along x
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Forward Difference. </summary>
	timer.Start();
	/// TODO: Call the kernel here
	dxt(d_src, d_dst, dimx, dimy, dimz, DDT_FORWARD);
	timer.Stop();
	printf("Forward Differentiation along x: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_fdxt.bin", h_dst, total*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Inverse Difference. </summary>
	timer.Start();
	dxt(d_src, d_dst, dimx, dimy, dimz, DDT_INVERSE);
	timer.Stop();
	printf("Inverse Differentiation along x: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_idxt.bin", h_dst, total*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Laplacian Difference. </summary>
	timer.Start();
	dxt(d_src, d_dst, dimx, dimy, dimz, DDT_LAPLACIAN);
	timer.Stop();
	printf("Laplace Differentiation along x: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_ldxt.bin", h_dst, total*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Along y
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Forward Difference. </summary>
	timer.Start();
	/// TODO: Call the kernel here
	dyt(d_src, d_dst, dimx, dimy, dimz, DDT_FORWARD);
	timer.Stop();
	printf("Forward Differentiation along y: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_fdyt.bin", h_dst, total*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Inverse Difference. </summary>
	timer.Start();
	dyt(d_src, d_dst, dimx, dimy, dimz, DDT_INVERSE);
	timer.Stop();
	printf("Inverse Differentiation along y: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_idyt.bin", h_dst, total*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Laplacian Difference. </summary>
	timer.Start();
	dyt(d_src, d_dst, dimx, dimy, dimz, DDT_LAPLACIAN);
	timer.Stop();
	printf("Laplace Differentiation along y: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_ldyt.bin", h_dst, total*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Along z
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Forward Difference. </summary>
	timer.Start();
	/// TODO: Call the kernel here
	dzt(d_src, d_dst, dimx, dimy, dimz, DDT_FORWARD);
	timer.Stop();
	printf("Forward Differentiation along z: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_fdzt.bin", h_dst, total*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Inverse Difference. </summary>
	timer.Start();
	dzt(d_src, d_dst, dimx, dimy, dimz, DDT_INVERSE);
	timer.Stop();
	printf("Inverse Differentiation along z: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_idzt.bin", h_dst, total*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Laplacian Difference. </summary>
	timer.Start();
	dzt(d_src, d_dst, dimx, dimy, dimz, DDT_LAPLACIAN);
	timer.Stop();
	printf("Laplace Differentiation along z: %4.4f ms\n", timer.Elapsed());
	// Copy spectrum to host and save
	cudaMemcpy(h_dst, d_dst, total*sizeof(float2), cudaMemcpyDeviceToHost);
	checkWriteFile("lenna_128x128x256_ldzt.bin", h_dst, total*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	return 0;
}