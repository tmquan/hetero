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
/// Mirror effect, acts like Neumann Boundary Condition
#define at(x, y, z, dimx, dimy, dimz) (clamp(z, 0, dimz-1)*dimy*dimx		\
									+clamp(y, 0, dimy-1)*dimx				\
									+clamp(x, 0, dimx-1))				
// -----------------------------------------------------------------------------------
__global__
void __warmup(float *src, float *dst, int dimx, int dimy, int dimz)
{
    //3D global index
	int3 index_3d = make_int3(
		blockIdx.x*blockDim.x+threadIdx.x,
		blockIdx.y*blockDim.y+threadIdx.y,
		blockIdx.z*blockDim.z+threadIdx.z);
	
	//Check valid indices
	if (index_3d.x >= dimx || index_3d.y >= dimy || index_3d.z >= dimz)
		return;
	
	//
	dst[at(index_3d.x, index_3d.y, index_3d.z, dimx, dimy, dimz)]
	=  	src[at(index_3d.x, index_3d.y, index_3d.z, dimx, dimy, dimz)];
}
// -----------------------------------------------------------------------------------
void warmup(float *src, float *dst, int dimx, int dimy, int dimz)
{
	dim3 numBlocks((dimx/8 + ((dimx%8)?1:0)),
				(dimy/8 + ((dimy%8)?1:0)),
				(dimz/8 + ((dimz%8)?1:0)) );
	dim3 numThreads(8, 8, 8);
	__warmup<<<numBlocks, numThreads>>>(src, dst, dimx, dimy, dimz);
}
// -----------------------------------------------------------------------------------
__global__
void __heatflow(float *src, float *dst, int dimx, int dimy, int dimz)
{
	//3D global index
	int3 index_3d = make_int3(
		blockIdx.x*blockDim.x+threadIdx.x,
		blockIdx.y*blockDim.y+threadIdx.y,
		blockIdx.z*blockDim.z+threadIdx.z);
	
	//Check valid indices
	if (index_3d.x >= dimx || index_3d.y >= dimy || index_3d.z >= dimz)
		return;
	//
    int index_1d = index_3d.z * dimy * dimx +
                   index_3d.y * dimx +
                   index_3d.x;
	//
    float tmp = index_1d * 0.001f; // Prevent optimization
    
    float a, b, c, d, e, f, result;
    for(int k=0; k<4000000000; k++)
    {   
        a = src[at(index_3d.x+1, index_3d.y+0, index_3d.z+0, dimx, dimy, dimz)];
        b = src[at(index_3d.x-1, index_3d.y+0, index_3d.z+0, dimx, dimy, dimz)];
        
        c = src[at(index_3d.x+0, index_3d.y+1, index_3d.z+0, dimx, dimy, dimz)];
        d = src[at(index_3d.x+0, index_3d.y-1, index_3d.z+0, dimx, dimy, dimz)];
        
        e = src[at(index_3d.x+0, index_3d.y+0, index_3d.z+1, dimx, dimy, dimz)];
        f = src[at(index_3d.x+0, index_3d.y+0, index_3d.z-1, dimx, dimy, dimz)];
        
        
        result = (a+b+c+d+e+f)/6.0f;
        
        a = src[at(index_3d.x+2, index_3d.y+0, index_3d.z+0, dimx, dimy, dimz)];
        b = src[at(index_3d.x-2, index_3d.y+0, index_3d.z+0, dimx, dimy, dimz)];
        
        c = src[at(index_3d.x+0, index_3d.y+2, index_3d.z+0, dimx, dimy, dimz)];
        d = src[at(index_3d.x+0, index_3d.y-2, index_3d.z+0, dimx, dimy, dimz)];
        
        e = src[at(index_3d.x+0, index_3d.y+0, index_3d.z+2, dimx, dimy, dimz)];
        f = src[at(index_3d.x+0, index_3d.y+0, index_3d.z-2, dimx, dimy, dimz)];
    }   
        
	dst[at(index_3d.x, index_3d.y, index_3d.z, dimx, dimy, dimz)] = result;
	// dst[at(index_3d.x, index_3d.y, index_3d.z, dimx, dimy, dimz)]
	// =  	(src[at(index_3d.x+1, index_3d.y+0, index_3d.z+0, dimx, dimy, dimz)] +
		// src[at(index_3d.x-1, index_3d.y+0, index_3d.z+0, dimx, dimy, dimz)] +
		
		// src[at(index_3d.x+0, index_3d.y+1, index_3d.z+0, dimx, dimy, dimz)] +
		// src[at(index_3d.x+0, index_3d.y-1, index_3d.z+0, dimx, dimy, dimz)] +
		
		// src[at(index_3d.x+0, index_3d.y+0, index_3d.z+1, dimx, dimy, dimz)] +
		// src[at(index_3d.x+0, index_3d.y+0, index_3d.z-1, dimx, dimy, dimz)]) / 6.0f;
}
// -----------------------------------------------------------------------------------
void heatflow(float *src, float *dst, int dimx, int dimy, int dimz)
{
	dim3 numBlocks((dimx/8 + ((dimx%8)?1:0)),
				(dimy/8 + ((dimy%8)?1:0)),
				(dimz/8 + ((dimz%8)?1:0)) );
	dim3 numThreads(8, 8, 8);
	__heatflow<<<numBlocks, numThreads>>>(src, dst, dimx, dimy, dimz);
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
            // cudaDeviceEnablePeerAccess	(localRank % deviceCount, 0);	cudaCheckLastError();
            for(int d=0; d<deviceCount; d++)
            {
                if(d!=(localRank % deviceCount))
                {
                    cudaDeviceEnablePeerAccess	(d, 0);	cudaCheckLastError();
                }                
            }
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
	MPI_Request req;
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
		"{ h   |help      |      | print help message }"
		"{     |dimx      | 512  | Number of the columns }"
		"{     |dimy      | 512  | Number of the rows }"
		"{     |dimz      | 512  | Temporal resolution }"
		"{ n   |numLoops  | 10   | Temporal resolution }";
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
    const int numLoops  = cmd.get<int>("numLoops", false);
	// if(rank==master)	cmd.printParams();
	// if(rank==master)	cout << dimx << endl << dimy << endl << dimz << endl;
	//================================================================================
	//!!! Determine main problem size and data partition same as CUDA style
	dim3 procDim;
	dim3 knotDim;
	dim3 haloDim;

	haloDim.x = 0;
	haloDim.y = 0;
	haloDim.z = 1;	// Pad 1 
	
	procDim.x = dimx;
	procDim.y = dimy;
	procDim.z = dimz/numWorkers;	// We partition only along z
	if(numWorkers==1)
		procDim.z = dimz/numWorkers;	// We partition only along z
	else
	{
		if(rank==head) 			procDim.z = 1*haloDim.z + dimz/numWorkers + 0*haloDim.z;	// We partition only along z
		else if(rank==tail) 	procDim.z = 0*haloDim.z + dimz/numWorkers + 1*haloDim.z;	// We partition only along z
		else					procDim.z = 1*haloDim.z + dimz/numWorkers + 1*haloDim.z;	// We partition only along z
	}
	knotDim.x  = dimx/procDim.x + (dimx%procDim.x)?1:0;
	knotDim.y  = dimy/procDim.y + (dimy%procDim.y)?1:0;
	knotDim.z  = dimz/procDim.z + (dimz%procDim.z)?1:0;
	
	if(rank==head) 
	{
		cout << procDim.x << endl << procDim.y << endl << procDim.z << endl;
		cout << knotDim.x  << endl << knotDim.y  << endl << knotDim.z  << endl;
	}
	
	//================================================================================
	// Master node will handle source and destination data
	float *h_src, *h_dst;
	h_src = NULL;
	h_dst = NULL;
	

	int total 		= dimx * dimy * dimz;
	int validSize	= dimx * dimy * dimz / numWorkers; // Valid data range
	int haloSize    = dimx * dimy * haloDim.z;
					
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
	
	int procSize    = procDim.x*procDim.y*procDim.z;
	
	
	float *p_src, *p_dst;
	p_src = NULL;
	p_dst = NULL;
	//================================================================================
	MPI_Sync("");
	cout << "Allocating src memory at " << rank << endl;
	MPI_Sync("");
	//================================================================================
	p_src = new float[procSize];
	// if(numWorkers == 1)
		// p_src = new float[validSize];
	// else
	// {
		// if(rank==head)	 		p_src = new float[headSize];
		// else if (rank==tail)	p_src = new float[tailSize];
		// else 					p_src = new float[middleSize];
	// }
	//================================================================================
	MPI_Sync("Done");
	//================================================================================
	MPI_Sync("");
	cout << "Allocating dst memory at " << rank << endl;
	MPI_Sync("");
	//================================================================================
	p_dst = new float[procSize];
	// if(numWorkers == 1)
		// p_dst = new float[validSize];
	// else
	// {
		// if(rank==head)	 		p_dst = new float[headSize];
		// else if (rank==tail)	p_dst = new float[tailSize];
		// else 					p_dst = new float[middleSize];
	// }
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
			MPI_Isend(h_src,                                headSize, MPI_FLOAT, head, 0, MPI_COMM_WORLD, &request);
			// Send to tail
			MPI_Isend(h_src + tail*validSize - haloSize,    tailSize, MPI_FLOAT, tail, 0, MPI_COMM_WORLD, &request);	
			// Send to middle
			for(int mid=head+1; mid<tail; mid++)
				MPI_Isend(h_src + mid*validSize - haloSize, middleSize, MPI_FLOAT, mid, 0, MPI_COMM_WORLD, &request);
		}
	}
	
	// Receive data
	MPI_Recv(p_src, procSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status); 
	// if(numWorkers==1)
		// MPI_Recv(p_src, validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status);
	// else
	// {
		// // Send to head
		// if(rank==head) 		MPI_Recv(p_src, headSize,   MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status);
		// else if(rank==tail) 	MPI_Recv(p_src, tailSize,   MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status);
		// else					MPI_Recv(p_src, middleSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &status);
	// }
	MPI_Sync("Done");
	//================================================================================
	// Processing here, assume processed, copy directly form src to dst

	MPI_Sync("Processing the data");
	// Common pattern
	if(numWorkers==1)
		; // Adjust the size
	else
	{
		if(rank==head) 			; // Adjust the size
		else if(rank==tail) 	; // Adjust the size
		else					; // Adjust the size
	}
	// if(numWorkers==1)
		// memcpy(p_dst, p_src, validSize*sizeof(float));
	// else
	// {
		// // Send to head
		// if(rank==head) 			memcpy(p_dst, p_src, headSize*sizeof(float));
		// else if(rank==tail) 	memcpy(p_dst, p_src, tailSize*sizeof(float));
		// else					memcpy(p_dst, p_src, middleSize*sizeof(float));
	// }
	
	// Declare GPU memory
	float *d_src;
	cudaMalloc((void**)&d_src, (procSize)*sizeof(float));
	// if(numWorkers==1)
		// cudaMalloc((void**)&d_src, (validSize)*sizeof(float));
	// else
	// {
		// if(rank==head) 			cudaMalloc((void**)&d_src, (headSize)*sizeof(float));
		// else if(rank==tail) 	cudaMalloc((void**)&d_src, (tailSize)*sizeof(float));
		// else					cudaMalloc((void**)&d_src, (middleSize)*sizeof(float));
	// }
	
	float *d_dst;
	cudaMalloc((void**)&d_dst, (procSize)*sizeof(float));
	// if(numWorkers==1)
		// cudaMalloc((void**)&d_dst, (validSize)*sizeof(float));
	// else
	// {
		// if(rank==head) 		cudaMalloc((void**)&d_dst, (headSize)*sizeof(float));
		// else if(rank==tail) 	cudaMalloc((void**)&d_dst, (tailSize)*sizeof(float));
		// else					cudaMalloc((void**)&d_dst, (middleSize)*sizeof(float));
	// }
	MPI_Sync("");
	//================================================================================
	// Copy to GPU memory
	cudaMemcpy(d_src, p_src, (procSize)*sizeof(float), cudaMemcpyHostToDevice);
	// if(numWorkers==1)
		// cudaMemcpy(d_src, p_src, (validSize)*sizeof(float), cudaMemcpyHostToDevice);
	// else
	// {
		// if(rank==head) 		cudaMemcpy(d_src, p_src, (headSize)*sizeof(float), cudaMemcpyHostToDevice);
		// else if(rank==tail) 	cudaMemcpy(d_src, p_src, (tailSize)*sizeof(float), cudaMemcpyHostToDevice);
		// else					cudaMemcpy(d_src, p_src, (middleSize)*sizeof(float), cudaMemcpyHostToDevice);
	// }
	MPI_Sync("");
    //================================================================================
    // for(int loop=0; loop<numLoops; loop++)
    // {
        // cudaDeviceSynchronize();		cudaCheckLastError();
        // MPI_Sync("");
		// // Launch the kernel
		// warmup(d_src, d_dst, procDim.x, procDim.y, procDim.z);
		// // if(numWorkers==1)
			// // heatflow(d_src, d_dst, procDim.x, procDim.y, procDim.z);
		// // else
		// // {
			// // if(rank==head) 		heatflow(d_src, d_dst, procDim.x, procDim.y, procDim.z+1*haloDim.z);
			// // else if(rank==tail) 	heatflow(d_src, d_dst, procDim.x, procDim.y, 1*haloDim.z+procDim.z);
			// // else					heatflow(d_src, d_dst, procDim.x, procDim.y, 1*haloDim.z+procDim.z+1*haloDim.z);
		// // }
	
		// // Device synchronize
		// cudaDeviceSynchronize();		cudaCheckLastError();
        // // Transfer the halo here
        // // Copy to right, tail cannot perform
        // //  // +-+-+---------+-+-+     +-+-+---------+-+-+     +-+-+---------+-+-+
        // // --> |R| | (i,j-1) |S| | --> |R| |  (i,j)  |S| | --> |R| | (i,j+1) |S| | -->
        // //  // +-+-+---------+-+-+     +-+-+---------+-+-+     +-+-+---------+-+-+
        // if(numWorkers==1)
            // ; // No need
        // else
        // {
            // if(rank<tail)	MPI_Isend(d_dst + procSize - 2*haloSize, haloSize, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &request);
            // if(rank>head)	MPI_Recv (d_dst, 						 haloSize, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);					
        // } 
        // MPI_Sync("Transfer to right for warming up");
        
        // // Copy to left, head cannot perform
        // // // +-+-+---------+-+-+     +-+-+---------+-+-+     +-+-+---------+-+-+
        // //<-- |X|S| (i,j-1) | |R| <-- |X|S|  (i,j)  | |R| <-- |X|S| (i,j+1) | |R| <--
        // // // +-+-+---------+-+-+     +-+-+---------+-+-+     +-+-+---------+-+-+
        // if(numWorkers==1)
            // ; // No need
        // else
        // {
            // if(rank>head)	MPI_Isend(d_dst + 1*haloSize, 			 haloSize, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &request);
            // if(rank<tail)	MPI_Recv (d_dst + procSize - 1*haloSize, haloSize, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);					
        // } 
        // MPI_Sync("Transfer to left for warming up");
        // cudaDeviceSynchronize();		cudaCheckLastError();
        // MPI_Sync("");
    // }
    //================================================================================
    cudaDeviceSynchronize();		cudaCheckLastError();
	MPI_Sync("");
	//================================================================================
    MPI_Request requests[2];
    MPI_Status statuses[2];
    double start = MPI_Wtime();
	// Launch the kernel
	for(int loop=0; loop<numLoops; loop++)
	{
        cudaDeviceSynchronize();		cudaCheckLastError();
        // MPI_Sync("");
		// Launch the kernel
		heatflow(d_src, d_dst, procDim.x, procDim.y, procDim.z);
		// if(numWorkers==1)
			// heatflow(d_src, d_dst, procDim.x, procDim.y, procDim.z);
		// else
		// {
			// if(rank==head) 		heatflow(d_src, d_dst, procDim.x, procDim.y, procDim.z+1*haloDim.z);
			// else if(rank==tail) 	heatflow(d_src, d_dst, procDim.x, procDim.y, 1*haloDim.z+procDim.z);
			// else					heatflow(d_src, d_dst, procDim.x, procDim.y, 1*haloDim.z+procDim.z+1*haloDim.z);
		// }
	
		// Device synchronize
		cudaDeviceSynchronize();		cudaCheckLastError();
		// MPI_Sync("Device Synchronization");	
		
		// Transfer the halo here
		// Copy to right, tail cannot perform
		//  // +-+-+---------+-+-+     +-+-+---------+-+-+     +-+-+---------+-+-+
		// --> |R| | (i,j-1) |S| | --> |R| |  (i,j)  |S| | --> |R| | (i,j+1) |S| | -->
		//  // +-+-+---------+-+-+     +-+-+---------+-+-+     +-+-+---------+-+-+
		if(numWorkers==1)
			; // No need
		else
		{
			if(rank<tail)	MPI_Isend(d_dst + procSize - 2*haloSize, haloSize, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &request);
			if(rank>head)	MPI_Recv (d_dst, 						 haloSize, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);					
			// if(rank>head)	MPI_Irecv(d_dst, 						 haloSize, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &(requests[0]));					
		} 
		MPI_Sync("Transfer to right");
		// MPI_WaitAll();
		// Copy to left, head cannot perform
		// // +-+-+---------+-+-+     +-+-+---------+-+-+     +-+-+---------+-+-+
		//<-- |X|S| (i,j-1) | |R| <-- |X|S|  (i,j)  | |R| <-- |X|S| (i,j+1) | |R| <--
		// // +-+-+---------+-+-+     +-+-+---------+-+-+     +-+-+---------+-+-+
		if(numWorkers==1)
			; // No need
		else
		{
			if(rank>head)	MPI_Isend(d_dst + 1*haloSize, 			 haloSize, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &request);
			if(rank<tail)	MPI_Recv (d_dst + procSize - 1*haloSize, haloSize, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);					
			// if(rank<tail)	MPI_Irecv(d_dst + procSize - 1*haloSize, haloSize, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &(requests[1]));					
		} 
		MPI_Sync("Transfer to left");
		// int count = 2;
        // MPI_Waitall(count, requests, statuses);
        // MPI_Barrier(MPI_COMM_WORLD);
        
		if(loop==(numLoops-1))	break;
		std::swap(d_src, d_dst);
	}
	// MPI_Sync("");
    //================================================================================
    double elapsed = MPI_Wtime() - start;
	if(rank == master)
		cout << "HeatFlow finish: " << endl
             << "dimx: " << dimx << endl
             << "dimy: " << dimy << endl
             << "dimz: " << dimz << endl
             << "numProcess(es): " << numWorkers << endl
             << "Execution time (s): " << elapsed << endl;
	MPI_Sync("Done");
	//================================================================================
    // Copy to CPU memory
	cudaMemcpy(p_dst, d_dst, (procSize)*sizeof(float), cudaMemcpyDeviceToHost);
	// if(numWorkers==1)
		// cudaMemcpy(p_dst, d_dst, (validSize)*sizeof(float), cudaMemcpyDeviceToHost);
	// else
	// {
		// if(rank==head) 		cudaMemcpy(p_dst, d_dst, (headSize)*sizeof(float), cudaMemcpyDeviceToHost);
		// else if(rank==tail) 	cudaMemcpy(p_dst, d_dst, (tailSize)*sizeof(float), cudaMemcpyDeviceToHost);
		// else					cudaMemcpy(p_dst, d_dst, (middleSize)*sizeof(float), cudaMemcpyDeviceToHost);
	// }
	//================================================================================
	// Calculate the golden result
	float *h_src_ref = NULL;
	float *h_dst_ref = NULL;
	if(rank==master)	
	{
		h_src_ref = new float[total];
		h_dst_ref = new float[total];
		memcpy(h_src_ref, h_src, total*sizeof(float));
		
		for(int loop=0; loop<numLoops; loop++)
		{
			for(int z=0; z<dimz; z++)
				for(int y=0; y<dimy; y++)
					for(int x=0; x<dimx; x++)
						h_dst_ref[at(x, y, z, dimx, dimy, dimz)] =	(h_src_ref[at(x+1, y+0, z+0, dimx, dimy, dimz)] +
																	h_src_ref[at(x-1, y+0, z+0, dimx, dimy, dimz)] +
																	h_src_ref[at(x+0, y+1, z+0, dimx, dimy, dimz)] +
																	h_src_ref[at(x+0, y-1, z+0, dimx, dimy, dimz)] +
																	h_src_ref[at(x+0, y+0, z+1, dimx, dimy, dimz)] +
																	h_src_ref[at(x+0, y+0, z-1, dimx, dimy, dimz)]) /6.0f;
			if(loop==(numLoops-1))	break;
			std::swap(h_src_ref, h_dst_ref);
		}
	}
	//================================================================================
	// Gathering the data
	MPI_Sync("Master is gathering the data");
	/// Send data
	if(numWorkers==1)
		MPI_Isend(p_dst, validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &request);
	else
	{
		// Send to head
		if(rank==head) 			MPI_Isend(p_dst, 				validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &request);
		else if(rank==tail) 	MPI_Isend(p_dst + haloSize, 	validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &request);
		else					MPI_Isend(p_dst + haloSize, 	validSize, MPI_FLOAT, master, 0, MPI_COMM_WORLD, &request);
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
			if(h_dst_ref[k] != h_dst[k])
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
