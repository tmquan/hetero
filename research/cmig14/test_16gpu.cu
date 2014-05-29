#include <omp.h>
#include <mpi.h>
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

#define MAX_DEVICES_PER_NODE 8
#define MAX_DEVICES_PER_PROC 8
// #define MPI_MASTER 3
// #define WORKER 
int main(int argc, char* argv[])
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
	/// Total problem size
	// const int dTotal = dimx*dimy*dimz;
	const int nTotal = dimx*dimy*dimz*dimn;

	/// Declare and allocate the host memories
	float2 *h_spec;
	float2 *h_full;
	float2 *h_mask;
	float2 *h_zero;
	float2 *h_dest;

	
	// /// Read data from file and store to memory
	// checkReadFile(spec.c_str(), h_spec, nTotal*sizeof(float2));
	// checkReadFile(mask.c_str(), h_mask, nTotal*sizeof(float2));
	////////////////////////////////////////////////////////////////////////////
	int localRank;
	char *localRankStr = NULL;
	
	
	if ((localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL)
	{
		localRank = atoi(localRankStr);		
		printf("Local rank %02d\n", localRank);
	}

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	cudaSetDevice(localRank % deviceCount);
	cudaDeviceReset();
	checkLastError();
	
	int rank, size;
	char name[MPI_MAX_PROCESSOR_NAME];
	int leng;
	
	MPI::Init(argc, argv);
	// MPI_Init(&argc, &argv);
	MPI::Status status;
	

	
	size = MPI::COMM_WORLD.Get_size();
	// MPI_Comm_size(MPI_COMM_WORLD, &size);
	rank = MPI::COMM_WORLD.Get_rank();
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	MPI::Get_processor_name(name, leng);
	// MPI_Get_processor_name(name, &leng);
	printf("This is process %02d out of %02d from %s\n", rank, size, name);
	// MPI::COMM_WORLD.Barrier();
	
	////////////////////////////////////////////////////////////////////////////
	
	int worker;
	int numMasters = 1;
	int numWorkers = size-1;
	
	const int master = size-1;
	const int head = 0;
	const int tail = size-2;
	
	//Print out the paramemters
	if(rank == master)
	{
		////////////////////////////////////////////////////////////////////////////
		printf("Master process is reading file from %d out of %d\n", rank, size);
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
		
		/// Read data from file and store to memory
		h_spec  = new float2[nTotal];
		h_full  = new float2[nTotal];
		h_mask  = new float2[nTotal];
		h_zero  = new float2[nTotal];
		h_dest  = new float2[nTotal];
		/// Waste of CPU Memory, but who cares?
		checkReadFile(spec.c_str(), h_spec, nTotal*sizeof(float2));
		checkReadFile(mask.c_str(), h_mask, nTotal*sizeof(float2));
	}		
	MPI::COMM_WORLD.Barrier();
	// // MPI_Barrier(MPI_COMM_WORLD);
	// ////////////////////////////////////////////////////////////////////////////
	// // cudaSetDevice(rank&7);  //8 GPUs for each node
	// // cudaDeviceReset(); 
	
	if(rank == master)
		printf("This is master %02d\n", rank);
	else if(rank == head)
		printf("This is head %02d\n", rank);
	else if(rank == tail)
		printf("This is tail %02d\n", rank);
	else //if(rank == link)
		printf("This is link %02d\n", rank);
	MPI::COMM_WORLD.Barrier();
	////////////////////////////////////////////////////////////////////////////
	/// <summary>	Reconstruct the full data	</summary>
	if(rank == master)
		printf("Starting Full Reconstruction.. \n");
	
	/// Copy the spectrum from host to device
	int hTems = 6;
	int mTems = dimz/numWorkers;
	int dTems;
	int hTotal;
	int mTotal;
	int	dTotal;
	

	
	hTotal = dimx*dimy*hTems;
	mTotal = dimx*dimy*mTems;
	

	//Template
	if(rank == master)
	{}
	else if(rank == head)
	{}
	else if(rank == tail)
	{}
	else //if(rank == link)
	{}
	MPI::COMM_WORLD.Barrier();
		
	// Determine temporal resolution for each process
	if(rank != master)
	{
		if(numWorkers == 1)
		{
			dTems  = mTems;
			dTotal = dimx*dimy*dTems;
		}
		else
		{
			if(rank == head)
			{
				dTems  = hTems + mTems;
				dTotal = dimx*dimy*dTems;
			}
			else if(rank == tail)
			{
				dTems  = mTems + hTems;
				dTotal = dimx*dimy*dTems;
			}
			else
			{
				dTems  = hTems + mTems + hTems;
				dTotal = dimx*dimy*dTems;
			}
		}
	}
	

	
	
	//Allocate process's memory from host
	float2 *p_spec, *p_full;
	if(rank != master)
	{		
		p_spec = new float2[dTotal];
		p_full = new float2[dTotal];
	}

	MPI::COMM_WORLD.Barrier();
	
	MPI::Status stat;
	//Distribute kspace data from master process to worker process (including halos)
	if(rank == master)
	{
		printf("Master is sending kspace..\n");
		if(numWorkers == 1)
		{
			MPI::COMM_WORLD.Send(h_spec, 
				mTotal, 
				MPI::DOUBLE, 
				head, 
				0);
		}
		else
		{
			MPI::COMM_WORLD.Send(h_spec, 
				mTotal + hTotal, 
				MPI::DOUBLE, 
				head, 
				0);
			MPI::COMM_WORLD.Send(h_spec + tail*mTotal - hTotal, 
				hTotal + mTotal, 
				MPI::DOUBLE, 
				tail, 
				0);
			for(int link=1; link<tail; link++)
			{
				MPI::COMM_WORLD.Send(h_spec + link*mTotal - hTotal, 
					hTotal + mTotal + hTotal, 
					MPI::DOUBLE, 
					link, 
					0);
			}
		}
		printf("Master sent kspace..\n");
	}
	else 
	{
		MPI::COMM_WORLD.Recv(p_spec, 
			dTotal,	// mTotal + hTotal from head
			MPI::DOUBLE, 
			master, 
			0, stat);
	}
	MPI::COMM_WORLD.Barrier();
	
	
	//Set device
	if(rank != master)
	{
		cudaSetDevice(rank&7);
		cudaDeviceReset();
	}
	MPI::COMM_WORLD.Barrier();
	
	/// Create Fourier plan 2.5d. 
	cufftHandle plan;
	if(rank != master)
	{
		int topo[3]	= {dimx, dimy, dTems};
		cufftPlanMany(&plan,
				2,			//Dimensionality of the transform (1, 2, or 3)
				topo,		//Array of size rank, describing the size of each dimension
				NULL,
				1,			//Distance between two successive input elements in the innermost dimension
				dimy*dimx,  //Distance between the first element of two consecutive signals
				NULL,
				1,
				dimy*dimx,
				CUFFT_C2C,
				dTems);
	}
	MPI::COMM_WORLD.Barrier();
		
	/// Allocate device memory
	float2 *d_spec, *d_full;
	if(rank != master)
	{
		cudaMalloc((void**)&d_spec, (dTotal)*sizeof(float2));
		cudaMalloc((void**)&d_full, (dTotal)*sizeof(float2));
	}
	
	//Transfer data from host to device memory
	if(rank != master)
	{
		cudaMemcpyAsync(
			d_spec, 
			p_spec, 
			(dTotal)*sizeof(float2), cudaMemcpyDefault); //cudaMemcpyHostToDevice
		cudaDeviceSynchronize();
	}
	MPI::COMM_WORLD.Barrier();
	
	//Do the inverse Fourier transform and scale to image domain
	if(rank != master)
	{
		dft(d_spec, d_full, dimx, dimy, dTems, DFT_INVERSE, plan);
		scale(d_full, d_full, dimx, dimy, dTems, 1.0f/(dimx*dimy));
	}
	

	//Copy back to host memory of each process
	if(rank != master) //Do nothing
	{
		cudaMemcpyAsync(
			p_full,
			d_full, 
			(dTotal)*sizeof(float2), cudaMemcpyDefault); //cudaMemcpyDeviceToHost
		cudaDeviceSynchronize();
	}
	MPI::COMM_WORLD.Barrier();
	
	//Send the reconstructed full data to master process, send only the main data
	if(rank == master)
	{
		printf("Master is receiving full data..\n");
		if(numWorkers == 1)
		{
			MPI::COMM_WORLD.Recv(h_full, 
				mTotal, 
				MPI::DOUBLE, 
				head, 
				0);
		}
		else
		{
			MPI::COMM_WORLD.Recv(h_full, 
				mTotal, 
				MPI::DOUBLE, 
				head, 
				0, 
				stat);
			MPI::COMM_WORLD.Recv(h_full + tail*mTotal, 
				mTotal, 
				MPI::DOUBLE, 
				tail, 
				0, 
				stat);
			for(int link=1; link<tail; link++)
			{
				MPI::COMM_WORLD.Recv(h_full + link*mTotal, 
					mTotal, 
					MPI::DOUBLE, 
					link, 
					0, 
					stat);
			}
		}
		printf("Master received full data..\n");
	}
	else if(rank == head)
	{
		MPI::COMM_WORLD.Send(p_full, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	else if(rank == tail)
	{
		MPI::COMM_WORLD.Send(p_full+hTotal, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	else //if(rank == link)
	{
		MPI::COMM_WORLD.Send(p_full+hTotal, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	
	//Use master process to save the full reconstruction data to file
	if(rank == master)
	{
		checkWriteFile(full.c_str(), h_full, nTotal*sizeof(float2));
	}
	
	if(rank == master)
		printf("Full Reconstruction.. done\n");
	////////////////////////////////////////////////////////////////////////////
	/// <summary>	Check zero reconstruction. </summary>
	if(rank == master)
		printf("Starting Zero Reconstruction.. \n");
	
	float2 *p_mask, *p_fill, *p_zero;
	if(rank != master)
	{		
		p_mask = new float2[dTotal];
		p_fill = new float2[dTotal];
		p_zero = new float2[dTotal];
	}
	MPI::COMM_WORLD.Barrier();
	
	/// Allocate device memory
	float2 *d_mask, *d_fill, *d_zero;
	if(rank != master)
	{
		cudaMalloc((void**)&d_mask, (dTotal)*sizeof(float2));
		cudaMalloc((void**)&d_fill, (dTotal)*sizeof(float2));
		cudaMalloc((void**)&d_zero, (dTotal)*sizeof(float2));
	}
	MPI::COMM_WORLD.Barrier();
	
	//Distribute mask data from master process to worker process (including halos)
	if(rank == master)
	{
		printf("Master is sending mask..\n");
		if(numWorkers == 1)
		{
			MPI::COMM_WORLD.Send(h_mask, 
				mTotal, 
				MPI::DOUBLE, 
				head, 
				0);
		}
		else
		{
			MPI::COMM_WORLD.Send(h_mask, 
				mTotal + hTotal, 
				MPI::DOUBLE, 
				head, 
				0);
			MPI::COMM_WORLD.Send(h_mask + tail*mTotal - hTotal, 
				hTotal + mTotal, 
				MPI::DOUBLE, 
				tail, 
				0);
			for(int link=1; link<tail; link++)
			{
				MPI::COMM_WORLD.Send(h_mask + link*mTotal - hTotal, 
					hTotal + mTotal + hTotal, 
					MPI::DOUBLE, 
					link, 
					0);
			}
		}
		printf("Master sent mask..\n");
	}
	else 
	{
		MPI::COMM_WORLD.Recv(p_mask, 
			dTotal,// hTotal + mTotal + hTotal from normal link
			MPI::DOUBLE, 
			master, 
			0, stat);
	}
	MPI::COMM_WORLD.Barrier();
	
	//Transfer data from host to device memory
	if(rank != master)
	{
		cudaMemcpyAsync(
			d_mask, 
			p_mask, 
			(dTotal)*sizeof(float2), cudaMemcpyDefault); //cudaMemcpyHostToDevice
		cudaDeviceSynchronize();
	}
	MPI::COMM_WORLD.Barrier();
	if(rank != master)
	{
		/// Subsampling kspace
		mul(d_spec, d_mask, d_fill, dimx, dimy, dTems);
		
		/// Perform Inverse Fourier and Scale
		dft(d_fill, d_zero, dimx, dimy, dTems, DFT_INVERSE, plan);
		scale(d_zero, d_zero, dimx, dimy, dTems, 1.0f/(dimx*dimy) );
		cudaDeviceSynchronize();
		checkLastError();
	}
	
	//Copy back to host memory of each process
	if(rank != master) //Do nothing
	{
		cudaMemcpyAsync(
			p_zero,
			d_zero, 
			(dTotal)*sizeof(float2), cudaMemcpyDefault); //cudaMemcpyDeviceToHost
		cudaDeviceSynchronize();
	}
	MPI::COMM_WORLD.Barrier();
	
	//Send the reconstructed zero data to master process, send only the main data
	if(rank == master)
	{
		printf("Master is receiving zero data..\n");
		if(numWorkers == 1)
		{
			MPI::COMM_WORLD.Recv(h_zero, 
				mTotal, 
				MPI::DOUBLE, 
				head, 
				0);
		}
		else
		{
			MPI::COMM_WORLD.Recv(h_zero, 
				mTotal, 
				MPI::DOUBLE, 
				head, 
				0, 
				stat);
			MPI::COMM_WORLD.Recv(h_zero + tail*mTotal, 
				mTotal, 
				MPI::DOUBLE, 
				tail, 
				0, 
				stat);
			for(int link=1; link<tail; link++)
			{
				MPI::COMM_WORLD.Recv(h_zero + link*mTotal, 
					mTotal, 
					MPI::DOUBLE, 
					link, 
					0, 
					stat);
			}
		}
		printf("Master received zero data..\n");
	}
	else if(rank == head)
	{
		MPI::COMM_WORLD.Send(p_zero, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	else if(rank == tail)
	{
		MPI::COMM_WORLD.Send(p_zero+hTotal, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	else //if(rank == link)
	{
		MPI::COMM_WORLD.Send(p_zero+hTotal, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	MPI::COMM_WORLD.Barrier();
	
	//Use master process to save the zero reconstruction data to file
	if(rank == master)
	{
		checkWriteFile(zero.c_str(), h_zero, nTotal*sizeof(float2));
	}
	if(rank == master)
		printf("Zero Reconstruction.. done\n");
	////////////////////////////////////////////////////////////////////////////
	/// <summary>	Reconstruct compressive sensing data	 </summary>
	float2 *p_dest;
	if(rank != master)
	{		
		p_dest = new float2[dTotal]; //Save the result
	}
	MPI::COMM_WORLD.Barrier();
	float2 *d_f;
	
	float2 *d_f0;
	float2 *d_ft;
		
	float2 *d_Ax;
	float2 *d_rhs;
	float2 *d_murf;
	float2 *d_Rft;
	float2 *d_Rf;
	
	float2 *d_R;
		
	// float2 *d_pu;	//Previous u
		
	float2 *d_cu;	//Current u
	
	float2 *d_x;	float2 *d_y;	float2 *d_z;	float2 *d_w;
	float2 *d_dx;	float2 *d_dy;	float2 *d_dz;	float2 *d_dw;
	float2 *d_lx;	float2 *d_ly;	float2 *d_lz;	float2 *d_lw;
	float2 *d_tx;	float2 *d_ty;	float2 *d_tz;	float2 *d_tw;
	float2 *d_bx;	float2 *d_by;	float2 *d_bz;	float2 *d_bw;
	float2 *d_xbx;	float2 *d_yby;	float2 *d_zbz;	float2 *d_wbw;
	float2 *d_dxbx;	float2 *d_dyby;	float2 *d_dzbz;	float2 *d_dwbw;
	
	cudaStream_t sTransferToPrev;	cudaStream_t sTransferToNext;
	cudaStream_t sTransfer;
	cudaEvent_t ePrev;	cudaEvent_t eNext;
	
	
	////////////////////////////////////////////////////////////////////////////
	if(rank == master)
		printf("Starting CS Reconstruction.. \n");
	
	if(rank != master)
	{
		cudaMalloc((void**)&d_f   , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_f0  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_ft  , dTotal*sizeof(float2));

		cudaMalloc((void**)&d_Ax  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_rhs , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_murf, dTotal*sizeof(float2));
		cudaMalloc((void**)&d_Rft , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_Rf  , dTotal*sizeof(float2));
		checkLastError();
		cudaMemset(d_Ax, 0, dTotal*sizeof(float2));
		cudaMemset(d_rhs, 0, dTotal*sizeof(float2));
		
		cudaMalloc((void**)&d_R   , dTotal*sizeof(float2));

		cudaMalloc((void**)&d_cu  , dTotal*sizeof(float2));

		cudaMalloc((void**)&d_x   , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_y   , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_z   , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_w   , dTotal*sizeof(float2));

		cudaMemset(d_x, 0, dTotal*sizeof(float2));
		cudaMemset(d_y, 0, dTotal*sizeof(float2));
		cudaMemset(d_z, 0, dTotal*sizeof(float2));
		cudaMemset(d_w, 0, dTotal*sizeof(float2));
		
		cudaMalloc((void**)&d_dx  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dy  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dz  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dw  , dTotal*sizeof(float2));
		
		cudaMemset(d_dx, 0, dTotal*sizeof(float2));
		cudaMemset(d_dy, 0, dTotal*sizeof(float2));
		cudaMemset(d_dz, 0, dTotal*sizeof(float2));
		cudaMemset(d_dw, 0, dTotal*sizeof(float2));
		
		cudaMalloc((void**)&d_lx  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_ly  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_lz  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_lw  , dTotal*sizeof(float2));
		
		cudaMemset(d_lx, 0, dTotal*sizeof(float2));
		cudaMemset(d_ly, 0, dTotal*sizeof(float2));
		cudaMemset(d_lz, 0, dTotal*sizeof(float2));
		cudaMemset(d_lw, 0, dTotal*sizeof(float2));

		cudaMalloc((void**)&d_tx  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_ty  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_tz  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_tw  , dTotal*sizeof(float2));

		cudaMemset(d_tx, 0, dTotal*sizeof(float2));
		cudaMemset(d_ty, 0, dTotal*sizeof(float2));
		cudaMemset(d_tz, 0, dTotal*sizeof(float2));
		cudaMemset(d_tw, 0, dTotal*sizeof(float2));
		
		cudaMalloc((void**)&d_bx  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_by  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_bz  , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_bw  , dTotal*sizeof(float2));

		cudaMemset(d_bx, 0, dTotal*sizeof(float2));
		cudaMemset(d_by, 0, dTotal*sizeof(float2));
		cudaMemset(d_bz, 0, dTotal*sizeof(float2));
		cudaMemset(d_bw, 0, dTotal*sizeof(float2));
		
		cudaMalloc((void**)&d_xbx , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_yby , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_zbz , dTotal*sizeof(float2));
		cudaMalloc((void**)&d_wbw , dTotal*sizeof(float2));
		
		cudaMemset(d_xbx, 0, dTotal*sizeof(float2));
		cudaMemset(d_yby, 0, dTotal*sizeof(float2));
		cudaMemset(d_zbz, 0, dTotal*sizeof(float2));
		cudaMemset(d_wbw, 0, dTotal*sizeof(float2));
		checkLastError();
		cudaMalloc((void**)&d_dxbx, dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dyby, dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dzbz, dTotal*sizeof(float2));
		cudaMalloc((void**)&d_dwbw, dTotal*sizeof(float2));
		checkLastError();
		cudaMemset(d_dxbx, 0, dTotal*sizeof(float2));
		cudaMemset(d_dyby, 0, dTotal*sizeof(float2));
		cudaMemset(d_dzbz, 0, dTotal*sizeof(float2));
		cudaMemset(d_dwbw, 0, dTotal*sizeof(float2));
		checkLastError();
		
		cudaStreamCreate(&sTransferToPrev);
		cudaStreamCreate(&sTransferToNext);
		cudaStreamCreate(&sTransfer);
		
		cudaEventCreateWithFlags(&ePrev,cudaEventDisableTiming);
		cudaEventCreateWithFlags(&eNext,cudaEventDisableTiming);
		checkLastError();
	}
	
	/// <summary> Copy kspace and mask. </summary>
	if(rank != master)
	{
		cudaMemcpyAsync(
				d_R, 
				d_mask, 
				dTotal*sizeof(float2), cudaMemcpyDefault); //cudaMemcpyDeviceToDevice
		cudaDeviceSynchronize();
	}
	MPI::COMM_WORLD.Barrier();
	
	if(rank != master)
	{
		/// Multiply the mask with the full kspace
		mul(d_spec, d_R, d_f, dimx, dimy, dTems);
		cudaDeviceSynchronize();
		/// Prepare the interpolation
		cudaMemcpyAsync(d_f0   , d_f, dTotal*sizeof(float2), cudaMemcpyDefault);
		cudaMemcpyAsync(d_ft   , d_f, dTotal*sizeof(float2), cudaMemcpyDefault);
		
		cudaDeviceSynchronize();
		checkLastError();
	}
	MPI::COMM_WORLD.Barrier();
	
	if(rank != master)
	{
		dft(d_f, d_cu, dimx, dimy, dTems, DFT_INVERSE, plan);
		scale(d_cu, d_cu, dimx, dimy, dTems, 1.0f/(dimx*dimy));
		scale(d_cu, d_murf, dimx, dimy, dTems, Mu);
		
		cudaDeviceSynchronize();
		checkLastError();
	}
	MPI::COMM_WORLD.Barrier();
	
	MPI::Request send_req1, recv_req1;
	MPI::Request send_req2, recv_req2;
	MPI::Request request;
	
	int prev_process = ((rank > head) && (rank != master)) ? (rank-1) : MPI_PROC_NULL;
	int next_process = ((rank < tail) && (rank != master)) ? (rank+1) : MPI_PROC_NULL;
	
	float2 *p_cu, *p_rhs; //Host pinned memory for data communication
	if(rank != master)
	{
		cudaHostAlloc((void**)&p_cu, 	dTotal*sizeof(float2), cudaHostAllocDefault);
		cudaHostAlloc((void**)&p_rhs, 	dTotal*sizeof(float2), cudaHostAllocDefault);
	}
	/// Run the reconstruction here
	double start = MPI::Wtime();
	if(rank != master)
	{
		bool isContinue = true;
		float  diff  = 0.0f;
		int iOuter = 0;
		int iInner = 0;
		int iLoops = 0;
		for(iOuter=0; iOuter<nOuter && isContinue ; iOuter++)
		{
			for(iInner=0; iInner<nInner && isContinue; iInner++)
			{
				/// Update Righ Hand Side term. 
				sub(d_x, d_bx, d_xbx, dimx, dimy, dTems);
				sub(d_y, d_by, d_yby, dimx, dimy, dTems);
				sub(d_z, d_bz, d_zbz, dimx, dimy, dTems);
				sub(d_w, d_bw, d_wbw, dimx, dimy, dTems);

				dxt(d_xbx, d_tx, dimx, dimy, dTems, DDT_INVERSE);
				dyt(d_yby, d_ty, dimx, dimy, dTems, DDT_INVERSE);
				dzt(d_zbz, d_tz, dimx, dimy, dTems, DDT_INVERSE);
				dwt(d_wbw, d_tw, dimx, dimy, dTems, DWT_INVERSE);
				
				scale(d_tx, d_tx, dimx, dimy, dTems, Lambda_w);
				scale(d_ty, d_ty, dimx, dimy, dTems, Lambda_w);
				scale(d_tz, d_tz, dimx, dimy, dTems, Gamma_w);
				scale(d_tw, d_tw, dimx, dimy, dTems, Omega_w);
				
				// If we have 1 GPU, we dont need to communicate
				if(numWorkers == 1)
				{
					add(d_tx, d_ty, d_tz, d_tw, d_murf, d_rhs, 
						dimx, dimy, dTems);
				}
				else
				{	/// Comunication here, update d_rhs, d_cu					
					// Prev Exterior
					add(
						d_tx, 
						d_ty, 
						d_tz, 
						d_tw, 
						d_murf, 
						d_rhs, 
						dimx, 
						dimy, 
						2*hTems);
					cudaStreamSynchronize(); 
					// Next Exterior
					add(
						d_tx + dTotal - 2*hTotal, 
						d_ty + dTotal - 2*hTotal, 
						d_tz + dTotal - 2*hTotal, 
						d_tw + dTotal - 2*hTotal, 
						d_murf + dTotal - 2*hTotal, 
						d_rhs + dTotal - 2*hTotal, 
						dimx, 
						dimy, 
						2*hTems);
					cudaStreamSynchronize();
					//Interior
					add(
						d_tx + 2*hTotal, 
						d_ty + 2*hTotal, 
						d_tz + 2*hTotal, 
						d_tw + 2*hTotal, 
						d_murf + 2*hTotal, 
						d_rhs + 2*hTotal, 
						dimx, dimy, dTems - 4*hTems,
						sTransfer);
					
					///////////////////////////////////////////////////////////////
					/// Copy left and right halo to process's host memory
					// if(rank > head)
					// {
						// cudaMemcpyAsync(p_cu + hTotal, //dst
							// d_cu + hTotal,// src
							// 1*hTotal*sizeof(float2),
							// cudaMemcpyDeviceToHost); //Modify stream here
						// cudaMemcpyAsync(p_rhs + hTotal, //dst
							// d_rhs + hTotal,// src
							// 1*hTotal*sizeof(float2),
							// cudaMemcpyDeviceToHost); //Modify stream here
					// }
					// if(rank < tail)
					// {
						// cudaMemcpyAsync(p_cu + dTotal - 2*hTotal, //dst
							// d_cu +  dTotal - 2*hTotal,// src
							// 1*hTotal*sizeof(float2),
							// cudaMemcpyDeviceToHost); //Modify stream here
						// cudaMemcpyAsync(p_rhs +  dTotal - 2*hTotal, //dst
							// d_rhs +  dTotal - 2*hTotal,// src
							// 1*hTotal*sizeof(float2),
							// cudaMemcpyDeviceToHost); //Modify stream here
					// }
					// cudaStreamSynchronize(0);
					// ////////////////////////////////////////////////////////////////////////////////	
					// if(rank > head)
					// {
						// MPI::COMM_WORLD.Isend(p_cu  + hTotal, 1*hTotal, MPI::DOUBLE, prev_process, 0);
						// MPI::COMM_WORLD.Isend(p_rhs + hTotal, 1*hTotal, MPI::DOUBLE, prev_process, 0);
					// }
					// if(rank < tail)
					// {
						// MPI::COMM_WORLD.Isend(p_cu  + dTotal - 2*hTotal, 1*hTotal, MPI::DOUBLE, next_process, 0);
						// MPI::COMM_WORLD.Isend(p_rhs + dTotal - 2*hTotal, 1*hTotal, MPI::DOUBLE, next_process, 0);
					// }
					// ////////////////////////////////////////////////////////////////////////////////
					// if(rank < tail)
					// {
						// MPI::COMM_WORLD.Recv(p_cu   + dTotal - 1*hTotal, 1*hTotal, MPI::DOUBLE, next_process, 0, stat);
						// MPI::COMM_WORLD.Recv(p_rhs  + dTotal - 1*hTotal, 1*hTotal, MPI::DOUBLE, next_process, 0, stat);
					// }
					// if(rank > head)
					// {
						// MPI::COMM_WORLD.Recv(p_cu , 1*hTotal, MPI::DOUBLE, prev_process, 0, stat);
						// MPI::COMM_WORLD.Recv(p_rhs, 1*hTotal, MPI::DOUBLE, prev_process, 0, stat);
					// }
					// ////////////////////////////////////////////////////////////////////////////////
					// if(rank < tail)
					// {
						// cudaMemcpyAsync(d_cu + dTotal - 1*hTotal, //dst
							// p_cu +  dTotal - 1*hTotal,// src
							// 1*hTotal*sizeof(float2),
							// cudaMemcpyHostToDevice);
						// cudaMemcpyAsync(d_rhs + dTotal - 1*hTotal, //dst
							// p_rhs +  dTotal - 2*hTotal,// src
							// 1*hTotal*sizeof(float2),
							// cudaMemcpyHostToDevice);
					// }
					// if(rank > head)
					// {
						// cudaMemcpyAsync(d_cu, //dst
							// p_cu,// src
							// 1*hTotal*sizeof(float2),
							// cudaMemcpyHostToDevice);
						// cudaMemcpyAsync(d_rhs, //dst
							// p_rhs,// src
							// 1*hTotal*sizeof(float2),
							// cudaMemcpyHostToDevice);
					// }
					// cudaDeviceSynchronize();
					////////////////////////////////////////////////////////////////////////////////	
					if(rank > head)
					{
						MPI::COMM_WORLD.Isend(d_cu  + hTotal, 1*hTotal, MPI::DOUBLE, prev_process, 0);
						MPI::COMM_WORLD.Isend(d_rhs + hTotal, 1*hTotal, MPI::DOUBLE, prev_process, 0);
					}
					if(rank < tail)
					{
						MPI::COMM_WORLD.Isend(d_cu  + dTotal - 2*hTotal, 1*hTotal, MPI::DOUBLE, next_process, 0);
						MPI::COMM_WORLD.Isend(d_rhs + dTotal - 2*hTotal, 1*hTotal, MPI::DOUBLE, next_process, 0);
					}
					////////////////////////////////////////////////////////////////////////////////
					if(rank < tail)
					{
						MPI::COMM_WORLD.Recv(d_cu   + dTotal - 1*hTotal, 1*hTotal, MPI::DOUBLE, next_process, 0, stat);
						MPI::COMM_WORLD.Recv(d_rhs  + dTotal - 1*hTotal, 1*hTotal, MPI::DOUBLE, next_process, 0, stat);
					}
					if(rank > head)
					{
						MPI::COMM_WORLD.Recv(d_cu , 1*hTotal, MPI::DOUBLE, prev_process, 0, stat);
						MPI::COMM_WORLD.Recv(d_rhs, 1*hTotal, MPI::DOUBLE, prev_process, 0, stat);
					}
				}
			
				
				///	Update u term.
				for(iLoops=0; iLoops<nLoops; iLoops++)
				{	
					dxt(d_cu, d_lx, dimx, dimy, dTems, DDT_LAPLACIAN);
					dyt(d_cu, d_ly, dimx, dimy, dTems, DDT_LAPLACIAN);
					dzt(d_cu, d_lz, dimx, dimy, dTems, DDT_LAPLACIAN);
					
					scale(d_lx, d_lx, dimx, dimy, dTems, Lambda_w);
					scale(d_ly, d_ly, dimx, dimy, dTems, Lambda_w);
					scale(d_lz, d_lz, dimx, dimy, dTems, Gamma_w);
					scale(d_cu, d_lw, dimx, dimy, dTems, Omega_w);
					
					dft(d_cu,  d_Ax, dimx, dimy, dTems, DFT_FORWARD, plan);
					
					mul(d_Ax, d_R,  d_Ax,  dimx, dimy, dTems);
					dft(d_Ax, d_Ax, dimx, dimy, dTems, DFT_INVERSE, plan);
					scale(d_Ax, d_Ax, dimx, dimy, dTems, 1.0f/(dimx*dimy)*Mu);
					
					add(d_lx, d_ly, d_lz, d_lw, d_Ax, d_Ax, 
						dimx, dimy, dTems);
					// add(d_lz, d_Ax, d_Ax, 
						// dimx, dimy, dTems);
					
					sub(d_rhs, d_Ax, d_Ax, dimx, dimy, dTems);
					
					scale(d_Ax, d_Ax, dimx, dimy, dTems, Ep);
					add(d_cu, d_Ax, d_cu, dimx, dimy, dTems);
				}	

	
					
				/// Update x, y, z. 
				dxt(d_cu, d_dx, dimx, dimy, dTems, DDT_FORWARD);
				dyt(d_cu, d_dy, dimx, dimy, dTems, DDT_FORWARD);
				dzt(d_cu, d_dz, dimx, dimy, dTems, DDT_FORWARD);
				dwt(d_cu, d_dw, dimx, dimy, dTems, DWT_FORWARD);
				
				add(d_dx, d_bx, d_dxbx, dimx, dimy, dTems);
				add(d_dy, d_by, d_dyby, dimx, dimy, dTems);
				add(d_dz, d_bz, d_dzbz, dimx, dimy, dTems);
				add(d_dw, d_bw, d_dwbw, dimx, dimy, dTems);
				
				shrink2(d_dxbx, d_dyby, d_x, d_y, dimx, dimy, dTems, Lambda_t);
				// shrink1(d_dxbx, d_x, dimx, dimy, dTems, Lambda_t);
				// shrink1(d_dyby, d_y, dimx, dimy, dTems, Lambda_t);
				shrink1(d_dzbz, d_z, dimx, dimy, dTems, Gamma_t);
				shrink1(d_dwbw, d_w, dimx, dimy, dTems, Omega_t);
				
				/// Update Bregman parameters. 
				sub(d_dxbx, d_x, d_bx, dimx, dimy, dTems);
				sub(d_dyby, d_y, d_by, dimx, dimy, dTems);
				sub(d_dzbz, d_z, d_bz, dimx, dimy, dTems);
				sub(d_dwbw, d_w, d_bw, dimx, dimy, dTems);	
			}
			/// Update Interpolation
			dft(d_cu, d_ft  , dimx , dimy, dTems, DFT_FORWARD, plan);
			mul(d_ft, d_R   , d_Rft, dimx, dimy, dTems);		
			add(d_f0, d_f   , d_f  , dimx, dimy, dTems);
			sub(d_f , d_Rft , d_f  , dimx, dimy, dTems);
			mul(d_f , d_R   , d_Rf , dimx, dimy, dTems);
			
			dft(d_Rf, d_murf, dimx , dimy, dTems, DFT_INVERSE, plan);
			scale(d_murf, d_murf, dimx, dimy, dTems, 1.0f/(dimx*dimy)*Mu);
		}
	}
	MPI::COMM_WORLD.Barrier();
	
	double elapsed = MPI::Wtime() - start;
	if(rank == master)
		printf("CS Reconstruction time: %4.3f seconds\n", elapsed);
	//Copy back to host memory of each process
	if(rank != master) //Do nothing
	{
		cudaMemcpyAsync(
			p_dest,
			d_cu, 
			(dTotal)*sizeof(float2), cudaMemcpyDefault); //cudaMemcpyDeviceToHost
		cudaDeviceSynchronize();
	}
	MPI::COMM_WORLD.Barrier();
	
	//Send the reconstructed zero data to master process, send only the main data
	if(rank == master)
	{
		printf("Master is receiving cs reconstructed data..\n");
		if(numWorkers == 1)
		{
			MPI::COMM_WORLD.Recv(h_dest, 
				mTotal, 
				MPI::DOUBLE, 
				head, 
				0);
		}
		else
		{
			MPI::COMM_WORLD.Recv(h_dest, 
				mTotal, 
				MPI::DOUBLE, 
				head, 
				0, 
				stat);
			MPI::COMM_WORLD.Recv(h_dest + tail*mTotal, 
				mTotal, 
				MPI::DOUBLE, 
				tail, 
				0, 
				stat);
			for(int link=1; link<tail; link++)
			{
				MPI::COMM_WORLD.Recv(h_dest + link*mTotal, 
					mTotal, 
					MPI::DOUBLE, 
					link, 
					0, 
					stat);
			}
		}
		printf("Master received cs reconstructed data..\n");
	}
	else if(rank == head)
	{
		MPI::COMM_WORLD.Send(p_dest, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	else if(rank == tail)
	{
		MPI::COMM_WORLD.Send(p_dest+hTotal, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	else //if(rank == link)
	{
		MPI::COMM_WORLD.Send(p_dest+hTotal, 
			mTotal, 
			MPI::DOUBLE, 
			master, 
			0);
	}
	MPI::COMM_WORLD.Barrier();
	
	//Use master process to save the zero reconstruction data to file
	if(rank == master)
	{
		checkWriteFile(dest.c_str(), h_dest, nTotal*sizeof(float2));
	}
	
	if(rank == master)
		printf("CS Reconstruction.. done\n");
	
	////////////////////////////////////////////////////////////////////////////
	//Clean up GPUs
	// if(rank != master)
	// {
		// cudaSetDevice(rank&7);
		// cudaDeviceReset();
	// }
	////////////////////////////////////////////////////////////////////////////
	
	MPI::Finalize();
	
	return 0;
}