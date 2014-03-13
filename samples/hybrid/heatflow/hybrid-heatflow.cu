#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <mpi.h>
#include <helper_math.h>
#include "cmdparser.hpp"
using namespace std;

// -----------------------------------------------------------------------------------
#define cudaCheckLastError() {                                          										\
	cudaError_t error = cudaGetLastError();                               										\
	int id; cudaGetDevice(&id);                                                     							\
	if(error != cudaSuccess) {                                                      							\
		printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",								\
			__FILE__,__LINE__, cudaGetErrorString(error), id);                      							\
		exit(EXIT_FAILURE);                                                         							\
	}                                                                               							\
}
// -----------------------------------------------------------------------------------
#define MPI_Sync(message) {                                 			        								\
	MPI_Barrier(MPI_COMM_WORLD);	                                                							\
	if(rank==master)		cout << "----------------------------------------------------------"<< endl;		\
	if(rank==master)		cout << message	<< endl;															\
	MPI_Barrier(MPI_COMM_WORLD);	                                               					 			\
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
	MPI_Status 	status;
	MPI_Request request;
	cout << "Hello World from rank " << rank 
		 << " out of " << size 
		 << " at " << name 
		 << endl;
							  

	//================================================================================
	// int master 		= size-1;
	// int worker;
	// int numMasters 	= 1;
	// int numWorkers 	= size-1;
	
	// int head = 0;
	// int tail = size-2;
	int master 		= 0;
	int worker;
	int numMasters 	= 1;
	int numWorkers 	= size;
	
	int head = 0;
	int tail = size-1;
	//================================================================================
	// Parsing the argument
	const char* key =
		"{ h   |help      |     | print help message }"
		"{     |dimx      | 64  | Number of the columns }"
		"{     |dimy      | 64  | Number of the rows }"
		"{     |dimz      | 64  | Temporal resolution }";
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
	
	haloDim.x = 0;
	haloDim.y = 0;
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
	

	int total 		= dimx * dimy * dimz;
	int validSize	= blockDim.x * blockDim.y * blockDim.z; // Valid data range
	int haloSize    = blockDim.x * blockDim.y * haloDim.z;
					  
	MPI_Sync("Allocating total memory at master");
	if(rank==master)
	{
		h_src = new float[total];
		
		h_dst = new float[total];
		for(int k=0; k<total; k++)
		{
			h_src[k] = (float)rand();
			h_dst[k] = 0;
		}
	}
	//================================================================================
	MPI_Sync("Done");
	//================================================================================
	// Worker or compute node will handle partially the data 
	// Head: validsize+haloSize
	// Middle: validsize+2*haloSize
	// Tail: validsize+haloSize
	int headSize    = validSize + 1*haloSize;
	int middleSize  = validSize + 2*haloSize;
	int tailSize    = validSize + 1*haloSize;
	
	float *p_src, *p_dst;
	p_src = NULL;
	p_dst = NULL;
	//================================================================================
	MPI_Sync("");
	cout << "Allocating src memory at " << rank << endl;
	MPI_Sync("");
	//================================================================================
	if(numWorkers == 1)
		p_src = new float[validSize];
	else
	{
		if(rank==head)	 		p_src = new float[headSize];
		else if (rank==tail)	p_src = new float[tailSize];
		else 					p_src = new float[middleSize];
	}
	//================================================================================
	MPI_Sync("Done");
	//================================================================================
	MPI_Sync("");
	cout << "Allocating dst memory at " << rank << endl;
	MPI_Sync("");
	//================================================================================
	if(numWorkers == 1)
		p_dst = new float[validSize];
	else
	{
		if(rank==head)	 		p_dst = new float[headSize];
		else if (rank==tail)	p_dst = new float[tailSize];
		else 					p_dst = new float[middleSize];
	}
	//================================================================================
	MPI_Sync("");
	cout << "Allocated  dst memory at " << rank << endl;
	MPI_Sync("Done");
	//================================================================================
	/// Start to distribute
	
	// Scatter the data
	MPI_Sync("Master is scattering the data");
	if(rank==master)	//Send
	{
		if(numWorkers==1)
			MPI_Isend(h_src, validSize, MPI_FLOAT, head, 0, MPI_COMM_WORLD, &request);
		else
		{
			// Send to head
			MPI_Isend(h_src, 							   	headSize, 	MPI_FLOAT, head, 0, MPI_COMM_WORLD, &request);
			// Send to tail
			MPI_Isend(h_src + tail*validSize - haloSize,    tailSize, 	MPI_FLOAT, tail, 0, MPI_COMM_WORLD, &request);	
			// Send to middle
			for(int mid=head+1; mid<tail; mid++)
				MPI_Isend(h_src + mid*validSize - haloSize, middleSize, MPI_FLOAT, mid, 0, MPI_COMM_WORLD, &request);
		}
	}
	
	// Receive data
	if(numWorkers==1)
		MPI_Recv(p_src, validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status);
	else
	{
		// Send to head
		if(rank==head) 			MPI_Recv(p_src, headSize,   MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status);
		else if(rank==tail) 	MPI_Recv(p_src, tailSize,   MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status);
		else					MPI_Recv(p_src, middleSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status);
	}
	MPI_Sync("Done");
	//================================================================================
	// Processing here, assume processed, copy directly form src to dst
	MPI_Sync("Processing the data");
	if(numWorkers==1)
		memcpy(p_dst, p_src, validSize*sizeof(float));
	else
	{
		// Send to head
		if(rank==head) 			memcpy(p_dst, p_src, headSize*sizeof(float));
		else if(rank==tail) 	memcpy(p_dst, p_src, tailSize*sizeof(float));
		else					memcpy(p_dst, p_src, middleSize*sizeof(float));
	}
	MPI_Sync("Done");
	//================================================================================
	// Gathering the data
	MPI_Sync("Master is gathering the data");
	/// Send data
	if(numWorkers==1)
		MPI_Isend(p_dst, validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &request);
	else
	{
		// Send to head
		if(rank==head) 			MPI_Isend(p_src, 				validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &request);
		else if(rank==tail) 	MPI_Isend(p_src + haloSize, 	validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &request);
		else					MPI_Isend(p_src + haloSize, 	validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &request);
	}
	/// Receive data
	if(rank==master)
	{
		if(numWorkers==1)
			MPI_Recv(h_dst, validSize, MPI_FLOAT, head, 0, MPI_COMM_WORLD, &status);
		else
		{
			// Send to head
			MPI_Recv(h_dst, 				    validSize, 	MPI_FLOAT, head, 0, MPI_COMM_WORLD, &status);
			// Send to tail
			MPI_Recv(h_dst + tail*validSize,    validSize, 	MPI_FLOAT, tail, 0, MPI_COMM_WORLD, &status);	
			// Send to middle
			for(int mid=head+1; mid<tail; mid++)
				MPI_Recv(h_dst + mid*validSize, validSize, 	MPI_FLOAT, mid,  0, MPI_COMM_WORLD, &status);
		}
	}
	MPI_Sync("Done");
	//================================================================================
	// check
	MPI_Sync("Master is checking the correctness");
	if(rank==master)	
	{
		for(int k=0; k<total; k++)
		{
			if(h_src[k] != h_dst[k])
			{
				cout << "Do not match at " << k << endl;
				goto cleanup;
			}
		}
		cout << "Matched!!!" << endl; 
		cleanup:
	}
	MPI_Sync("Done");
	//================================================================================
	// Finalize to join all of the MPI processes and terminate the program
	MPI_Finalize();
	return 0;
}
