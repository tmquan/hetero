#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <mpi.h>
#include <helper_math.h>
#include "cmdparser.hpp"
using namespace std;

// -----------------------------------------------------------------------------------
#define cudaCheckLastError() {                                          			\
	cudaError_t error = cudaGetLastError();                               			\
	int id; cudaGetDevice(&id);                                                     \
	if(error != cudaSuccess) {                                                      \
		printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",	\
			__FILE__,__LINE__, cudaGetErrorString(error), id);                      \
		exit(EXIT_FAILURE);                                                         \
	}                                                                               \
}
// -----------------------------------------------------------------------------------
int main (int argc, char *argv[])
{
	
	//================================================================================
	// To set the GPU using cudaSetDevice, we must set before launching MPI_Init
	// Determine the MPI local rank per node is doable either in OpenMPI or MVAPICH2
	int   localRank;
	char *localRankStr = NULL;
	//================================================================================
	// Investigate the number of GPUs per node.
	int deviceCount = 0;
	localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
	if (localRankStr != NULL)
	{
		localRank = atoi(localRankStr);		
		cudaGetDeviceCount(&deviceCount);
		// cudaCheckLastError();	//Don't put this line
		// printf("There are %02d device(s) at local process %02d\n", 
			// deviceCount, localRank);
		cout << "There are " << deviceCount 
			 <<	" device(s) at local process " 
			 << endl;
		if(deviceCount>0)
		{
			cudaSetDevice(localRank % deviceCount);	cudaCheckLastError();
			cudaDeviceReset();	cudaCheckLastError();
		}
	}
	//================================================================================
	// Information to control the MPI process
	// We have totally n processes, index from 0 to n-1
	// master process is indexed at n-1, totally 1
	// worker processes are indexed from 0 to n-2, totally (n-1)
	// the head process is indexed at 0
	// the tail process is indexed at (n-2)
	int size, rank;
	char name[MPI_MAX_PROCESSOR_NAME];
	int length;
		
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	
	cout << "Hello World from rank " << rank 
		 << " out of " << size 
		 << " at " << name 
		 << endl;
							  

	//================================================================================
	int master 		= size-1;
	int worker;
	int numMasters 	= 1;
	int numWorkers 	= size-1;
	
	int head = 0;
	int tail = size-2;
	
	//================================================================================
	// Parsing the argument
	const char* key =
		"{ h   |help      |       | print help message }"
		"{     |dimx      | 1024  | Number of the columns }"
		"{     |dimy      | 1024  | Number of the rows }"
		"{     |dimz      | 1024  | Temporal resolution }";
	CommandLineParser cmd(argc, argv, key);
	// if(rank==master)
	// if (argc == 1)
	// {
		// cout << "Usage: " << argv[0] << " [options]" << endl;
		// cout << "Avaible options:" << endl;
		// cmd.printParams();
		// return 0;
	// }
	//================================================================================
	const int dimx    	= cmd.get<int>("dimx", false); //default value has been provide
	const int dimy    	= cmd.get<int>("dimy", false);
	const int dimz    	= cmd.get<int>("dimz", false);
	// if(rank==master)	cmd.printParams();
	// if(rank==master)	cout << dimx << endl << dimy << endl << dimz << endl;
	//================================================================================
	// Determine main problem size and data partition same as CUDA style
	dim3 blockDim;
	dim3 gridDim;
	dim3 haloDim;

	blockDim.x = dimx;
	blockDim.y = dimy;
	blockDim.z = dimz/numWorkers;	// We partition only along z
	
	gridDim.x  = dimx/blockDim.x + (dimx%blockDim.x)?1:0;
	gridDim.y  = dimy/blockDim.y + (dimy%blockDim.y)?1:0;
	gridDim.z  = dimz/blockDim.z + (dimz%blockDim.z)?1:0;
	
	haloDim.x = 1;
	haloDim.y = 1;
	haloDim.z = 1;	// Pad 1 
	
	
	if(rank==head) 
	{
		cout << blockDim.x << endl << blockDim.y << endl << blockDim.z << endl;
		cout << gridDim.x  << endl << gridDim.y  << endl << gridDim.z  << endl;
	}
	
	//================================================================================
	// Master node will handle source and destination data
	float *h_src, *h_dst;
	h_src = NULL;
	h_dst = NULL;
	//!!! TODO: Preprocess, pad as mirror boundary
	int total 	= (dimx+2*haloDim.x) * 
				  (dimy+2*haloDim.y) *  
				  (dimz+2*haloDim.z);
				  
	int partial = (blockDim.x+2*haloDim.x) * 
				  (blockDim.y+2*haloDim.y) * 
				  (blockDim.z+2*haloDim.z);
				  
	int ghost   = (dimx+2*haloDim.x) * 
				  (dimy+2*haloDim.y) *  
				  (1*haloDim.z);
				  
	if(rank==master)
	{
		h_src = new float[total];
		h_dst = new float[total];
	}
	
	// Worker or compute node will handle partially the data + 2 halo data
	
	float *p_src, *p_dst;
	p_src = NULL;
	p_dst = NULL;
	
	int3 shared_index_3d{0, 0, 0};
	int  shared_index_1d = 0;
	
	int3 global_index_3d{0, 0, 0};
	int  global_index_1d = 0;
	// Memory allocation
	if(rank!=master)
	{
		p_src = new float[partial];
		p_dst = new float[partial];
		
		// int numTrials =  (((blockDim.x+2*haloDim.x) *  (blockDim.y+2*haloDim.y) *  (blockDim.z+2*haloDim.z)) /
						  // ((blockDim.x+0*haloDim.x) *  (blockDim.y+0*haloDim.y) *  (blockDim.z+0*haloDim.z)))	+ 
						 // ((((blockDim.x+2*haloDim.x) *  (blockDim.y+2*haloDim.y) *  (blockDim.z+2*haloDim.z)) %
						  // ((blockDim.x+0*haloDim.x) *  (blockDim.y+0*haloDim.y) *  (blockDim.z+0*haloDim.z)))?0:1);
		// cout << numTrials << endl;
		// // #pragma omp parallel 
		// for(blockIdx.z=0; blockIdx.z<gridDim.z; blockIdx.z++)
		// {
			// // #pragma omp parallel 
			// for(blockIdx.y=0; blockIdx.y<gridDim.y; blockIdx.y++)
			// {
				// // #pragma omp parallel 
				// for(blockIdx.x=0; blockIdx.x<gridDim.x; blockIdx.x++)
				// {
					// // #pragma omp parallel 
					// for(threadIdx.z=0; threadIdx.z<blockDim.z; threadIdx.z++)
					// {
						// // #pragma omp parallel 
						// for(threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y++)
						// {
							// // #pragma omp parallel 
							// for(threadIdx.x=0; threadIdx.x<blockDim.x; threadIdx.x++)
							// {
								
								// for(trial=0; trial<numTrials; trial++)
								// {
									// shared_index_1d 	= threadIdx.z * blockDim.y * blockDim.x +
														  // threadIdx.y * blockDim.x + 
														  // threadIdx.x +
														  // blockDim.x  * blockDim.y * blockDim.z * trial;  // Next number of loading
									// shared_index_3d		= make_int3((shared_index_1d % ((blockDim.y+2*haloDim.y) * (blockDim.x+2*haloDim.x))) % (blockDim.x+2*haloDim.x),
																	// (shared_index_1d % ((blockDim.y+2*haloDim.y) * (blockDim.x+2*haloDim.x))) / (blockDim.x+2*haloDim.x),
																	// (shared_index_1d / ((blockDim.y+2*haloDim.y) * (blockDim.x+2*haloDim.x))) );
									// global_index_3d		= make_int3(blockIdx.x * blockDim.x + shared_index_3d.x - haloDim.x,
																	// blockIdx.y * blockDim.y + shared_index_3d.y - haloDim.y,
																	// blockIdx.z * blockDim.z + shared_index_3d.z - haloDim.z);
									// global_index_1d 	= global_index_3d.z * dimy * dimx + 
														  // global_index_3d.y * dimx + 
														  // global_index_3d.x;
									// if (shared_index_3d.z < (blockDim.z + 2*haloDim.z)) 
									// {
										// if (global_index_3d.z >= 0 && global_index_3d.z < dimz && 
											// global_index_3d.y >= 0 && global_index_3d.y < dimy &&
											// global_index_3d.x >= 0 && global_index_3d.x < dimx )	
											// sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = src[global_index_1d];
										// else
											// sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = -1;
									// }
									// __syncthreads();
								// }
							// }
						// }
					// }
				// }
			// }
		// }// End data retrieve
	}
	
	/// Start to distribute
	if(rank==master)
	{
		for(int processIdx=head; processIdx<tail; processIdx++)
		{
			MPI_Send(h_src + processIdx*mTotal - hTotal, 
			// MPI::COMM_WORLD.Send(h_src + link*mTotal - hTotal, 
				// hTotal + mTotal + hTotal, 
				// MPI::DOUBLE, 
				// link, 
				// 0);
		}
	}
	// Finalize to join all of the MPI processes and terminate the program
	MPI_Finalize();
	return 0;
}
