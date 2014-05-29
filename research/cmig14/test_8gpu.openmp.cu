#include <omp.h>
#include <cuda.h>
#include <cufft.h>
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
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "cmdparser.hpp"
#include "utility.hpp"
#include "timer.hpp"
#include "add.hpp"
#include "sub.hpp"
#include "mul.hpp"
#include "ddt.hpp"
#include "dft.hpp"
#include "dwt.hpp"
#include "shrink.hpp"

using namespace std;
using namespace csmri;
__host__ __device__
void swap(float2* src, float2* dst)
{
	float2* tmp = src;
	src = dst;
	dst = tmp;
}
#define MAX_DEVICES 8
int main(int argc, const char* argv[])
{
		const char* key =
		"{ h   |help      |       | print help message }"
		"{     |spec      |       | Binary file of spectrum (kspace) - input }"
		"{     |full      |       | Binary file of full reconstruction}"
		"{     |mask      |       | Binary file of mask - input}"
		"{     |zero      |       | Binary file of zero filling reconstruction}"
		"{     |dest      |       | Binary file of reconstruction}"
		"{     |dimx      |       | Number of the columns }"
		"{     |dimy      |       | Number of the rows }"
		"{     |dimz      |       | Temporal resolution }"
		"{     |dimn      |       | Number of slices }"
		"{     |devs      | 1     | Number of GPUs }"
		"{     |Mu        | 0.100 | Weight of Interpolation }"
		"{     |Lambda_w  | 0.005 | Weight of Lambda }"
		"{     |Lambda_t  | 1.000 | Threshold of Lambda }"	
		"{     |Gamma_w   | 0.200 | Weight of Gamma }"
		"{     |Gamma_t   | 1.000 | Threshold of Gamma }"
		"{     |Omega_w   | 0.600 | Weight of Omega }"
		"{     |Omega_t   | 1.000 | Threshold of Omega}"
		"{     |Epsilon   | 0.700 | Epsilon of Richardson loop}"
		"{     |nOuter    | 4     | Number of Outer loops}"
		"{     |nInner    | 8     | Number of Inner loops}"
		"{     |nLoops    | 4     | Number of Richardson loops}";
		
	CommandLineParser cmd(argc, argv, key);
	if (argc == 1)
	{
		cout << "Usage: " << argv[0] << " [options]" << endl;
		cout << "Avaible options:" << endl;
		cmd.printParams();
		return 0;
	}
	
	// cmd.printParams();
	
	string spec 		= cmd.get<string>("spec", true);
	string full  		= cmd.get<string>("full", true);
	string mask  		= cmd.get<string>("mask", true);
	string zero  		= cmd.get<string>("zero", true);
	string dest			= cmd.get<string>("dest", true);
	////////////////////////////////////////////////////////////////////////////
	const int dimx    	= cmd.get<int>("dimx", true);
	const int dimy    	= cmd.get<int>("dimy", true);
	const int dimz    	= cmd.get<int>("dimz", true);
	const int dimn    	= cmd.get<int>("dimn", true);
	const int devs    	= cmd.get<int>("devs", true);
	////////////////////////////////////////////////////////////////////////////
	float Mu			= cmd.get<float>("Mu", true);
	float Lambda_w 		= cmd.get<float>("Lambda_w", true);
	float Lambda_t  	= cmd.get<float>("Lambda_t", true);
	float Gamma_w  		= cmd.get<float>("Gamma_w", true);
	float Gamma_t   	= cmd.get<float>("Gamma_t", true);
	float Omega_w		= cmd.get<float>("Omega_w", true);
	float Omega_t		= cmd.get<float>("Omega_t", true);
	float Ep	  		= cmd.get<float>("Epsilon", true);
	////////////////////////////////////////////////////////////////////////////
	int nOuter	  		= cmd.get<int>("nOuter", true);
	int nInner	  		= cmd.get<int>("nInner", true);
	int nLoops 	  		= cmd.get<int>("nLoops", true);
	////////////////////////////////////////////////////////////////////////////
	/// Print out the parameters
	cout << spec << endl;
	cout << mask << endl;
	cout << dest << endl;
	printf("Size: %dx%dx%d\n", dimx, dimy, dimz);
	printf("Number of GPUs: %d\n", devs);
	
	printf("Mu     : %4.4f\n", Mu);
	printf("Lambda : %4.4f\t\t%4.4f\n", Lambda_w, Lambda_t);
	printf("Gamma  : %4.4f\t\t%4.4f\n", Gamma_w,  Gamma_t);
	printf("Omega  : %4.4f\t\t%4.4f\n", Omega_w,  Omega_t);
	printf("Epsilon: %4.4f\n", Ep);
	
	printf("Number of loops: %dx%dx%d\n", nOuter, nInner, nLoops);
	////////////////////////////////////////////////////////////////////////////
	/// Total problem size
	const int dTotal = dimx*dimy*dimz;
	const int nTotal = dimx*dimy*dimz*dimn;
	
	/// Declare and allocate the host memories
	float2 *h_spec;
	float2 *h_full;
	float2 *h_mask;
	float2 *h_zero;
	float2 *h_dest;
	
	h_spec  = new float2[nTotal];
	h_full  = new float2[nTotal];
	h_mask  = new float2[nTotal];
	h_zero  = new float2[nTotal];
	h_dest  = new float2[nTotal];
	
	/// Read data from file and store to memory
	checkReadFile(spec.c_str(), h_spec, nTotal*sizeof(float2));
	checkReadFile(mask.c_str(), h_mask, nTotal*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaDeviceReset();
		cudaDeviceSynchronize();
		
		checkLastError();
	}
	////////////////////////////////////////////////////////////////////////////
	int rank[3] = {dimx, dimy, dimz};

	/// Create Fourier plan 2.5d. 
	cufftHandle plan[MAX_DEVICES];
	// cudaDeviceReset();
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);		
		cufftPlanMany(&plan[d],
			2,			//Dimensionality of the transform (1, 2, or 3)
			rank,		//Array of size rank, describing the size of each dimension
			NULL,
			1,			//Distance between two successive input elements in the innermost dimension
			dimy*dimx,  //Distance between the first element of two consecutive signals
			NULL,
			1,
			dimy*dimx,
			CUFFT_C2C,
			dimz);
		
		checkLastError();
	}
	
	/// Declare and allocate the device memories
	float2 *d_spec[MAX_DEVICES];
	float2 *d_full[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_spec[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_full[d], dTotal*sizeof(float2));
		
		cudaDeviceSynchronize();
		checkLastError();
	}

	
	////////////////////////////////////////////////////////////////////////////
	/// <summary>	Reconstruct the full data	</summary>
	/// Copy the spectrum from host to device
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMemcpyAsync(
				d_spec[d], 
				h_spec + d*dTotal, 
				dTotal*sizeof(float2), cudaMemcpyDefault);
		cudaDeviceSynchronize();	
		checkLastError();
	}
	/// Perform Inverse Fourier and Scale
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		dft(d_spec[d], d_full[d], dimx, dimy, dimz, DFT_INVERSE, plan[d]);
		scale(d_full[d], d_full[d], dimx, dimy, dimz, 1.0f/(dimx*dimy) );
		cudaDeviceSynchronize();
		checkLastError();
	}
	/// Copy the spectrum from device to host
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMemcpyAsync(
				h_full + d*dTotal, 
				d_full[d], 
				dTotal*sizeof(float2), cudaMemcpyDefault);
		cudaDeviceSynchronize();
		checkLastError();
	}
	/// Write the full reconstruction to binary file
	checkWriteFile(full.c_str(), h_full, nTotal*sizeof(float2));
	
	float2 *d_mask[MAX_DEVICES];
	float2 *d_fill[MAX_DEVICES];
	float2 *d_zero[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_mask[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_fill[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_zero[d], dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	//////////////////////////////////////////////////////////////////////////////
	/// <summary>	Reconstruct the zero filling data	</summary>
	/// Copy the mask from host to device
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		cudaMemcpyAsync(
				d_mask[d], 
				h_mask + d*dTotal, 
				dTotal*sizeof(float2), cudaMemcpyDefault);
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		/// Subsampling kspace
		mul(d_spec[d], d_mask[d], d_fill[d], dimx, dimy, dimz);
		
		/// Perform Inverse Fourier and Scale
		dft(d_fill[d], d_zero[d], dimx, dimy, dimz, DFT_INVERSE, plan[d]);
		scale(d_zero[d], d_zero[d], dimx, dimy, dimz, 1.0f/(dimx*dimy) );
		cudaDeviceSynchronize();
		checkLastError();
	}
	/// Copy the spectrum from device to host
	// cudaMemcpy(h_zero, d_zero, nTotal*sizeof(float2), cudaMemcpyDeviceToHost);
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		cudaMemcpyAsync(
				h_zero + d*dTotal, 
				d_zero[d], 
				dTotal*sizeof(float2), cudaMemcpyDefault);
		cudaDeviceSynchronize();
		checkLastError();
	}
	/// Write the zero reconstruction to binary file
	checkWriteFile(zero.c_str(), h_zero, nTotal*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////
	/// <summary>	Reconstruct the compressive sensing data	</summary>
	/// <summary>	Reserve Memory for the auxillary variables. </summary>

	float2 *d_f[MAX_DEVICES], *d_f0[MAX_DEVICES], *d_ft[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_f[d],  dTotal*sizeof(float2));
		cudaMalloc((void**)&d_f0[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_ft[d], dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}	
	
	float2 *d_Ax[MAX_DEVICES], *d_rhs[MAX_DEVICES], *d_murf[MAX_DEVICES], *d_Rft[MAX_DEVICES], *d_Rf[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_Ax[d]  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_rhs[d] , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_murf[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_Rft[d] , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_Rf[d]  , dTotal*sizeof(float2));
	
		cudaMemset(d_Ax[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_R[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_R[d], dTotal*sizeof(float2));
		cudaMemset(d_R[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_cu[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_cu[d], dTotal*sizeof(float2));
		cudaMemset(d_cu[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_x[MAX_DEVICES], *d_y[MAX_DEVICES], *d_z[MAX_DEVICES], *d_w[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_cu[d], dTotal*sizeof(float2));
		
		cudaMalloc((void**)&d_x[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_y[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_z[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_w[d], dTotal*sizeof(float2));
		
		cudaMemset(d_x[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_y[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_z[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_w[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_dx[MAX_DEVICES], *d_dy[MAX_DEVICES], *d_dz[MAX_DEVICES], *d_dw[MAX_DEVICES];
	
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_dx[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dy[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dz[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dw[d], dTotal*sizeof(float2));
		
		cudaMemset(d_dx[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_dy[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_dz[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_dw[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_lx[MAX_DEVICES], *d_ly[MAX_DEVICES], *d_lz[MAX_DEVICES], *d_lw[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		cudaMalloc((void**)&d_lx[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_ly[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_lz[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_lw[d], dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_tx[MAX_DEVICES], *d_ty[MAX_DEVICES], *d_tz[MAX_DEVICES], *d_tw[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_tx[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_ty[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_tz[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_tw[d], dTotal*sizeof(float2));
		
		cudaMemset(d_tx[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_ty[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_tz[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_tw[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_bx[MAX_DEVICES], *d_by[MAX_DEVICES], *d_bz[MAX_DEVICES], *d_bw[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		cudaMalloc((void**)&d_bx[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_by[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_bz[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_bw[d], dTotal*sizeof(float2));
		
		cudaMemset(d_bx[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_by[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_bz[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_bw[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_xbx[MAX_DEVICES], *d_yby[MAX_DEVICES], *d_zbz[MAX_DEVICES], *d_wbw[MAX_DEVICES];
	
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_xbx[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_yby[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_zbz[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_wbw[d], dTotal*sizeof(float2));
		
		cudaMemset(d_xbx[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_yby[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_zbz[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_wbw[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	float2 *d_dxbx[MAX_DEVICES], *d_dyby[MAX_DEVICES], *d_dzbz[MAX_DEVICES], *d_dwbw[MAX_DEVICES];
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		cudaMalloc((void**)&d_dxbx[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dyby[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dzbz[d], dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dwbw[d], dTotal*sizeof(float2));
		
		cudaMemset(d_dxbx[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_dyby[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_dzbz[d], 0, dTotal*sizeof(float2));
		cudaMemset(d_dwbw[d], 0, dTotal*sizeof(float2));
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	/// Copy kspace
	// cudaMemcpy(d_spec, h_spec, nTotal*sizeof(float2), cudaMemcpyHostToDevice);
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		cudaMemcpyAsync(
				d_spec[d], 
				h_spec + d*dTotal, 
				dTotal*sizeof(float2), cudaMemcpyDefault);
		cudaDeviceSynchronize();
		checkLastError();
	}
	/// Copy mask
	// cudaMemcpy(d_R   , h_mask, nTotal*sizeof(float2), cudaMemcpyHostToDevice);
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		cudaMemcpyAsync(
				d_R[d], 
				h_mask + d*dTotal, 
				dTotal*sizeof(float2), cudaMemcpyDefault);
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		/// Multiply the mask with the full kspace
		mul(d_spec[d], d_R[d], d_f[d], dimx, dimy, dimz);
		cudaDeviceSynchronize();
		/// Prepare the interpolation
		cudaMemcpyAsync(d_f0[d]   , d_f[d], dTotal*sizeof(float2), cudaMemcpyDefault);
		cudaMemcpyAsync(d_ft[d]   , d_f[d], dTotal*sizeof(float2), cudaMemcpyDefault);
		
		cudaDeviceSynchronize();
		checkLastError();
	}
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);	
		dft(d_f[d], d_cu[d], dimx, dimy, dimz, DFT_INVERSE, plan[d]);
		scale(d_cu[d], d_cu[d], dimx, dimy, dimz, 1.0f/(dimx*dimy));
		scale(d_cu[d], d_murf[d], dimx, dimy, dimz, Mu);
		
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	bool isContinue = true;
	float  diff  = 0.0f;
	int iOuter = 0;
	int iInner = 0;
	int iLoops = 0;
	/// Start the iterative method
	double start=omp_get_wtime();
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		for(iOuter=0; iOuter<nOuter && isContinue ; iOuter++)
		{
			for(iInner=0; iInner<nInner && isContinue; iInner++)
			{
				// #pragma omp parallel num_threads(devs) //private(id)
				// {
					// int d = omp_get_thread_num();
					// cudaSetDevice(d);
					
					/// Update Righ Hand Side term. 
					sub(d_x[d], d_bx[d], d_xbx[d], dimx, dimy, dimz);
					sub(d_y[d], d_by[d], d_yby[d], dimx, dimy, dimz);
					sub(d_z[d], d_bz[d], d_zbz[d], dimx, dimy, dimz);
					sub(d_w[d], d_bw[d], d_wbw[d], dimx, dimy, dimz);

					dxt(d_xbx[d], d_tx[d], dimx, dimy, dimz, DDT_INVERSE);
					dyt(d_yby[d], d_ty[d], dimx, dimy, dimz, DDT_INVERSE);
					dzt(d_zbz[d], d_tz[d], dimx, dimy, dimz, DDT_INVERSE);
					dwt(d_wbw[d], d_tw[d], dimx, dimy, dimz, DWT_INVERSE);
					
					scale(d_tx[d], d_tx[d], dimx, dimy, dimz, Lambda_w);
					scale(d_ty[d], d_ty[d], dimx, dimy, dimz, Lambda_w);
					scale(d_tz[d], d_tz[d], dimx, dimy, dimz, Gamma_w);
					scale(d_tw[d], d_tw[d], dimx, dimy, dimz, Omega_w);
					
					add(d_tx[d], d_ty[d], d_tz[d], d_tw[d], d_murf[d], d_rhs[d], 
						dimx, dimy, dimz);
					// add(d_tz[d], d_murf[d], d_rhs[d], 
						// dimx, dimy, dimz);
				// }
				///	Update u term.
				for(iLoops=0; iLoops<nLoops; iLoops++)
				{
					// #pragma omp parallel num_threads(devs) //private(id)
					// {
						// int d = omp_get_thread_num();
						// cudaSetDevice(d);
						
						dxt(d_cu[d], d_lx[d], dimx, dimy, dimz, DDT_LAPLACIAN);
						dyt(d_cu[d], d_ly[d], dimx, dimy, dimz, DDT_LAPLACIAN);
						dzt(d_cu[d], d_lz[d], dimx, dimy, dimz, DDT_LAPLACIAN);
						
						scale(d_lx[d], d_lx[d], dimx, dimy, dimz, Lambda_w);
						scale(d_ly[d], d_ly[d], dimx, dimy, dimz, Lambda_w);
						scale(d_lz[d], d_lz[d], dimx, dimy, dimz, Gamma_w);
						scale(d_cu[d], d_lw[d], dimx, dimy, dimz, Omega_w);
						
						dft(d_cu[d],  d_Ax[d], dimx, dimy, dimz, DFT_FORWARD, plan[d]);
						
						mul(d_Ax[d], d_R[d],  d_Ax[d],  dimx, dimy, dimz);
						dft(d_Ax[d], d_Ax[d], dimx, dimy, dimz, DFT_INVERSE, plan[d]);
						scale(d_Ax[d], d_Ax[d], dimx, dimy, dimz, 1.0f/(dimx*dimy)*Mu);
						
						add(d_lx[d], d_ly[d], d_lz[d], d_lw[d], d_Ax[d], d_Ax[d], 
							dimx, dimy, dimz);
						// add(d_lz[d], d_Ax[d], d_Ax[d], 
							// dimx, dimy, dimz);
						
						sub(d_rhs[d], d_Ax[d], d_Ax[d], dimx, dimy, dimz);
						
						scale(d_Ax[d], d_Ax[d], dimx, dimy, dimz, Ep);
						add(d_cu[d], d_Ax[d], d_cu[d], dimx, dimy, dimz);
					// }
				}	

				// #pragma omp parallel num_threads(devs) //private(id)
				// {
					// int d = omp_get_thread_num();
					// cudaSetDevice(d);
					
					/// Update x, y, z. 
					dxt(d_cu[d], d_dx[d], dimx, dimy, dimz, DDT_FORWARD);
					dyt(d_cu[d], d_dy[d], dimx, dimy, dimz, DDT_FORWARD);
					dzt(d_cu[d], d_dz[d], dimx, dimy, dimz, DDT_FORWARD);
					dwt(d_cu[d], d_dw[d], dimx, dimy, dimz, DWT_FORWARD);
					
					add(d_dx[d], d_bx[d], d_dxbx[d], dimx, dimy, dimz);
					add(d_dy[d], d_by[d], d_dyby[d], dimx, dimy, dimz);
					add(d_dz[d], d_bz[d], d_dzbz[d], dimx, dimy, dimz);
					add(d_dw[d], d_bw[d], d_dwbw[d], dimx, dimy, dimz);
					
					shrink2(d_dxbx[d], d_dyby[d], d_x[d], d_y[d], dimx, dimy, dimz, Lambda_t);
					// shrink1(d_dxbx, d_x, dimx, dimy, dimz, Lambda_t);
					// shrink1(d_dyby, d_y, dimx, dimy, dimz, Lambda_t);
					shrink1(d_dzbz[d], d_z[d], dimx, dimy, dimz, Gamma_t);
					shrink1(d_dwbw[d], d_w[d], dimx, dimy, dimz, Omega_t);
					
					/// Update Bregman parameters. 
					sub(d_dxbx[d], d_x[d], d_bx[d], dimx, dimy, dimz);
					sub(d_dyby[d], d_y[d], d_by[d], dimx, dimy, dimz);
					sub(d_dzbz[d], d_z[d], d_bz[d], dimx, dimy, dimz);
					sub(d_dwbw[d], d_w[d], d_bw[d], dimx, dimy, dimz);	
				// }
			}
			// #pragma omp parallel num_threads(devs) //private(id)
			// {
				int d = omp_get_thread_num();
				cudaSetDevice(d);
			
				/// Update Interpolation
				dft(d_cu[d], d_ft[d]  , dimx , dimy, dimz, DFT_FORWARD, plan[d]);
				mul(d_ft[d], d_R[d]   , d_Rft[d], dimx, dimy, dimz);		
				add(d_f0[d], d_f[d]   , d_f[d]  , dimx, dimy, dimz);
				sub(d_f[d] , d_Rft[d] , d_f[d]  , dimx, dimy, dimz);
				mul(d_f[d] , d_R[d]   , d_Rf[d] , dimx, dimy, dimz);
				
				dft(d_Rf[d], d_murf[d], dimx , dimy, dimz, DFT_INVERSE, plan[d]);
				scale(d_murf[d], d_murf[d], dimx, dimy, dimz, 1.0f/(dimx*dimy)*Mu);
			// }
		}
	}
	double end=omp_get_wtime();
	/// Copy the reconstruction from device to host
	//cudaMemcpy(h_dest, d_cu, nTotal*sizeof(float2), cudaMemcpyDeviceToHost);
	#pragma omp parallel num_threads(devs) //private(id)
	{
		int d = omp_get_thread_num();
		cudaSetDevice(d);
		
		cudaMemcpyAsync(
				h_dest + d*dTotal, 
				d_cu[d], 
				dTotal*sizeof(float2), cudaMemcpyDefault);
	}
	/// Write the full reconstruction to binary file
	checkWriteFile(dest.c_str(), h_dest, nTotal*sizeof(float2));
	cout << "CS Reconstruction: " <<  (end-start) << endl;
}