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

#include <hetero_cmdparser.hpp>
#include "utility.hpp"
#include "timer.hpp"
#include "add.hpp"
#include "sub.hpp"
#include "mul.hpp"
#include "ddt.hpp"
#include "dft.hpp"
#include "dwt.hpp"
#include "shrink.hpp"

// using namespace cv;
using namespace std;
using namespace csmri;

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
	
	cmd.printParams();
	
	string spec 		= cmd.get<string>("spec", true);
	string full  		= cmd.get<string>("full", true);
	string mask  		= cmd.get<string>("mask", true);
	string zero  		= cmd.get<string>("zero", true);
	string dest			= cmd.get<string>("dest", true);
	////////////////////////////////////////////////////////////////////////////
	const int dimx    	= cmd.get<int>("dimx", true);
	const int dimy    	= cmd.get<int>("dimy", true);
	const int dimz    	= cmd.get<int>("dimz", true);
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
	/// Device choosing and resetting
	cudaSetDevice(0); 	checkLastError();
	cudaDeviceReset();	checkLastError();
	////////////////////////////////////////////////////////////////////////////
	/// Total problem size
	const int nTotal = dimx*dimy*dimz;
	
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
	checkReadFile(spec, h_spec, nTotal*sizeof(float2));
	checkReadFile(mask, h_mask, nTotal*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////
	int rank[3] = {dimx, dimy, dimz};
	/// Create Fourier plan 2.5d. 
	cufftHandle plan;
	cufftPlanMany(&plan,
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
	
	/// Declare and allocate the device memories
	float2 *d_spec;
	float2 *d_full;
	cudaMalloc((void**)&d_spec, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_full, nTotal*sizeof(float2));		checkLastError();
	
	////////////////////////////////////////////////////////////////////////////
	/// <summary>	Reconstruct the full data	</summary>
	/// Copy the spectrum from host to device
	cudaMemcpy(d_spec, h_spec, nTotal*sizeof(float2), cudaMemcpyHostToDevice);
	checkLastError();
	
	/// Perform Inverse Fourier and Scale
	dft(d_spec, d_full, dimx, dimy, dimz, DFT_INVERSE, plan);
	scale(d_full, d_full, dimx, dimy, dimz, 1.0f/(dimx*dimy) );
	
	/// Copy the spectrum from device to host
	cudaMemcpy(h_full, d_full, nTotal*sizeof(float2), cudaMemcpyDeviceToHost);
	checkLastError();
	
	/// Write the full reconstruction to binary file
	checkWriteFile(full, h_full, nTotal*sizeof(float2));
	
	/// Declare and allocate the device memories
	float2 *d_mask;
	float2 *d_fill;
	float2 *d_zero;
	cudaMalloc((void**)&d_mask, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_fill, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_zero, nTotal*sizeof(float2));		checkLastError();
	////////////////////////////////////////////////////////////////////////////
	/// <summary>	Reconstruct the zero filling data	</summary>
	/// Copy the mask from host to device
	cudaMemcpy(d_mask, h_mask, nTotal*sizeof(float2), cudaMemcpyHostToDevice);
	checkLastError();
	
	/// Subsampling kspace
	mul(d_spec, d_mask, d_fill, dimx, dimy, dimz);
	
	/// Perform Inverse Fourier and Scale
	dft(d_fill, d_zero, dimx, dimy, dimz, DFT_INVERSE, plan);
	scale(d_zero, d_zero, dimx, dimy, dimz, 1.0f/(dimx*dimy) );
	
	/// Copy the spectrum from device to host
	cudaMemcpy(h_zero, d_zero, nTotal*sizeof(float2), cudaMemcpyDeviceToHost);
	checkLastError();
	
	/// Write the zero reconstruction to binary file
	checkWriteFile(zero, h_zero, nTotal*sizeof(float2));
	
	////////////////////////////////////////////////////////////////////////////
	/// <summary>	Reconstruct the compressive sensing data	</summary>
	/// <summary>	Reserve Memory for the auxillary variables. </summary>

	float2 *d_f, *d_f0, *d_ft;
	cudaMalloc((void**)&d_f,  nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_f0, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_ft, nTotal*sizeof(float2));		checkLastError();
	
	
	float2 *d_Ax, *d_rhs, *d_murf, *d_Rft, *d_Rf;
	cudaMalloc((void**)&d_Ax  , nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_rhs , nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_murf, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_Rft , nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_Rf  , nTotal*sizeof(float2));		checkLastError();
	
	cudaMemset(d_Ax, 0, nTotal*sizeof(float2));				checkLastError();
	
	float2 *d_R;
	cudaMalloc((void**)&d_R, nTotal*sizeof(float2));		checkLastError();
	cudaMemset(d_R, 0, nTotal*sizeof(float2));				checkLastError();
	
	float2 *d_cu;
	cudaMalloc((void**)&d_cu, nTotal*sizeof(float2));		checkLastError();
	cudaMemset(d_cu, 0, nTotal*sizeof(float2));				checkLastError();
		
	float2 *d_x, *d_y, *d_z, *d_w;
	cudaMalloc((void**)&d_cu, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_x, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_y, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_z, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_w, nTotal*sizeof(float2));		checkLastError();
	
	cudaMemset(d_x, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_y, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_z, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_w, 0, nTotal*sizeof(float2));				checkLastError();
	
	float2 *d_dx, *d_dy, *d_dz, *d_dw;
	cudaMalloc((void**)&d_dx, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_dy, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_dz, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_dw, nTotal*sizeof(float2));		checkLastError();
	
	cudaMemset(d_dx, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_dy, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_dz, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_dw, 0, nTotal*sizeof(float2));				checkLastError();
	
	float2 *d_lx, *d_ly, *d_lz, *d_lw;
	cudaMalloc((void**)&d_lx, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_ly, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_lz, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_lw, nTotal*sizeof(float2));		checkLastError();
	
	float2 *d_tx, *d_ty, *d_tz, *d_tw;
	cudaMalloc((void**)&d_tx, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_ty, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_tz, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_tw, nTotal*sizeof(float2));		checkLastError();
	
	cudaMemset(d_tx, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_ty, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_tz, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_tw, 0, nTotal*sizeof(float2));				checkLastError();
		
	float2 *d_bx, *d_by, *d_bz, *d_bw;
	cudaMalloc((void**)&d_bx, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_by, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_bz, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_bw, nTotal*sizeof(float2));		checkLastError();
	
	cudaMemset(d_bx, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_by, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_bz, 0, nTotal*sizeof(float2));				checkLastError();
	cudaMemset(d_bw, 0, nTotal*sizeof(float2));				checkLastError();
	
	float2 *d_xbx, *d_yby, *d_zbz, *d_wbw;
	cudaMalloc((void**)&d_xbx, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_yby, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_zbz, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_wbw, nTotal*sizeof(float2));		checkLastError();
	
	cudaMemset(d_xbx, 0, nTotal*sizeof(float2));			checkLastError();
	cudaMemset(d_yby, 0, nTotal*sizeof(float2));			checkLastError();
	cudaMemset(d_zbz, 0, nTotal*sizeof(float2));			checkLastError();
	cudaMemset(d_wbw, 0, nTotal*sizeof(float2));			checkLastError();
	
	float2 *d_dxbx, *d_dyby, *d_dzbz, *d_dwbw;
	cudaMalloc((void**)&d_dxbx, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_dyby, nTotal*sizeof(float2));		checkLastError();
	cudaMalloc((void**)&d_dzbz, nTotal*sizeof(float2));		checkLastError();		
	cudaMalloc((void**)&d_dwbw, nTotal*sizeof(float2));		checkLastError();
	
	cudaMemset(d_dxbx, 0, nTotal*sizeof(float2));			checkLastError();
	cudaMemset(d_dyby, 0, nTotal*sizeof(float2));			checkLastError();
	cudaMemset(d_dzbz, 0, nTotal*sizeof(float2));			checkLastError();
	cudaMemset(d_dwbw, 0, nTotal*sizeof(float2));			checkLastError();
	
	/// Copy kspace
	cudaMemcpy(d_spec, h_spec, nTotal*sizeof(float2), cudaMemcpyHostToDevice);
	checkLastError();
	
	/// Copy mask
	cudaMemcpy(d_R   , h_mask, nTotal*sizeof(float2), cudaMemcpyHostToDevice);
	checkLastError();
	
	/// Multiply the mask with the full kspace
	mul(d_spec, d_R, d_f, dimx, dimy, dimz);
	
	/// Prepare the interpolation
	cudaMemcpy(d_f0   , d_f, nTotal*sizeof(float2), cudaMemcpyDeviceToDevice);
	checkLastError();
	cudaMemcpy(d_ft   , d_f, nTotal*sizeof(float2), cudaMemcpyDeviceToDevice);
	checkLastError();
	
	
	dft(d_f, d_cu, dimx, dimy, dimz, DFT_INVERSE, plan);
	scale(d_cu, d_cu, dimx, dimy, dimz, 1.0f/(dimx*dimy));
	scale(d_cu, d_murf, dimx, dimy, dimz, Mu);
	
	
	
	bool isContinue = true;
	float  diff  = 0.0f;
	int iOuter = 0;
	int iInner = 0;
	int iLoops = 0;
	/// Start the iterative method
	GpuTimer timer;
	timer.Start();
	
	for(iOuter=0; iOuter<nOuter && isContinue ; iOuter++)
	{
		for(iInner=0; iInner<nInner && isContinue; iInner++)
		{
			/// Update Righ Hand Side term. 
			sub(d_x, d_bx, d_xbx, dimx, dimy, dimz);
			sub(d_y, d_by, d_yby, dimx, dimy, dimz);
			sub(d_z, d_bz, d_zbz, dimx, dimy, dimz);
			sub(d_w, d_bw, d_wbw, dimx, dimy, dimz);

			dxt(d_xbx, d_tx, dimx, dimy, dimz, DDT_INVERSE);
			dyt(d_yby, d_ty, dimx, dimy, dimz, DDT_INVERSE);
			dzt(d_zbz, d_tz, dimx, dimy, dimz, DDT_INVERSE);
			dwt(d_wbw, d_tw, dimx, dimy, dimz, DWT_INVERSE);
			
			scale(d_tx, d_tx, dimx, dimy, dimz, Lambda_w);
			scale(d_ty, d_ty, dimx, dimy, dimz, Lambda_w);
			scale(d_tz, d_tz, dimx, dimy, dimz, Gamma_w);
			scale(d_tw, d_tw, dimx, dimy, dimz, Omega_w);
			
			add(d_tx, d_ty, d_tz, d_tw, d_murf, d_rhs, dimx, dimy, dimz);
			
			///	Update u term.
			for(iLoops=0; iLoops<nLoops; iLoops++)
			{
				dxt(d_cu, d_lx, dimx, dimy, dimz, DDT_LAPLACIAN);
				dyt(d_cu, d_ly, dimx, dimy, dimz, DDT_LAPLACIAN);
				dzt(d_cu, d_lz, dimx, dimy, dimz, DDT_LAPLACIAN);
				
				scale(d_lx, d_lx, dimx, dimy, dimz, Lambda_w);
				scale(d_ly, d_ly, dimx, dimy, dimz, Lambda_w);
				scale(d_lz, d_lz, dimx, dimy, dimz, Gamma_w);
				scale(d_cu, d_lw, dimx, dimy, dimz, Omega_w);
				
				dft(d_cu,  d_Ax, dimx, dimy, dimz, DFT_FORWARD, plan);
				
				mul(d_Ax, d_R,  d_Ax,  dimx, dimy, dimz);
				dft(d_Ax, d_Ax, dimx, dimy, dimz, DFT_INVERSE, plan);
				scale(d_Ax, d_Ax, dimx, dimy, dimz, 1.0f/(dimx*dimy)*Mu);
				
				add(d_lx, d_ly, d_lz, d_lw, d_Ax, d_Ax, dimx, dimy, dimz);
				
				sub(d_rhs, d_Ax, d_Ax, dimx, dimy, dimz);
				
				scale(d_Ax, d_Ax, dimx, dimy, dimz, Ep);
				add(d_cu, d_Ax, d_cu, dimx, dimy, dimz);
			}	

			
			/// Update x, y, z. 
			dxt(d_cu, d_dx, dimx, dimy, dimz, DDT_FORWARD);
			dyt(d_cu, d_dy, dimx, dimy, dimz, DDT_FORWARD);
			dzt(d_cu, d_dz, dimx, dimy, dimz, DDT_FORWARD);
			dwt(d_cu, d_dw, dimx, dimy, dimz, DWT_FORWARD);
			
			add(d_dx, d_bx, d_dxbx, dimx, dimy, dimz);
			add(d_dy, d_by, d_dyby, dimx, dimy, dimz);
			add(d_dz, d_bz, d_dzbz, dimx, dimy, dimz);
			add(d_dw, d_bw, d_dwbw, dimx, dimy, dimz);
			
			shrink2(d_dxbx, d_dyby, d_x, d_y, dimx, dimy, dimz, Lambda_t);
			// shrink1(d_dxbx, d_x, dimx, dimy, dimz, Lambda_t);
			// shrink1(d_dyby, d_y, dimx, dimy, dimz, Lambda_t);
			shrink1(d_dzbz, d_z, dimx, dimy, dimz, Gamma_t);
			shrink1(d_dwbw, d_w, dimx, dimy, dimz, Omega_t);
			
			/// Update Bregman parameters. 
			sub(d_dxbx, d_x, d_bx, dimx, dimy, dimz);
			sub(d_dyby, d_y, d_by, dimx, dimy, dimz);
			sub(d_dzbz, d_z, d_bz, dimx, dimy, dimz);
			sub(d_dwbw, d_w, d_bw, dimx, dimy, dimz);					
		}
	
		/// Update Interpolation
		dft(d_cu, d_ft  , dimx , dimy, dimz, DFT_FORWARD, plan);
		mul(d_ft, d_R   , d_Rft, dimx, dimy, dimz);		
		add(d_f0, d_f   , d_f  , dimx, dimy, dimz);
		sub(d_f , d_Rft , d_f  , dimx, dimy, dimz);
		mul(d_f , d_R   , d_Rf , dimx, dimy, dimz);
		
		dft(d_Rf, d_murf, dimx , dimy, dimz, DFT_INVERSE, plan);
		scale(d_murf, d_murf, dimx, dimy, dimz, 1.0f/(dimx*dimy)*Mu);
		
	}
	timer.Stop();
	/// Copy the reconstruction from device to host
	cudaMemcpy(h_dest, d_cu, nTotal*sizeof(float2), cudaMemcpyDeviceToHost);
	checkLastError();
	/// Write the full reconstruction to binary file
	checkWriteFile(dest, h_dest, nTotal*sizeof(float2));
	cout << "CS Reconstruction: " << timer.Elapsed() << endl;
}