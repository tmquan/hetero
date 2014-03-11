#include <stdio.h>
#include <cuda.h>
#include <mpi.h>

#define cudaCheckLastError() {                                          			\
	cudaError_t error = cudaGetLastError();                               			\
	int id; cudaGetDevice(&id);                                                     \
	if(error != cudaSuccess) {                                                      \
		printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",	\
			__FILE__,__LINE__, cudaGetErrorString(error), id);                      \
		exit(EXIT_FAILURE);                                                         \
	}                                                                               \
}

int main (int argc, char *argv[])
{
	int localRank;
	char *localRankStr = NULL;
	////////////////////////////////////////////////////////////////////////////	
	int deviceCount = 0;
	localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
	if (localRankStr != NULL)
	{
		localRank = atoi(localRankStr);		
		cudaGetDeviceCount(&deviceCount);
		// cudaCheckLastError();	//Don't put this line
		printf("There are %02d device(s) at local process %02d\n", 
			deviceCount, localRank);
		if(deviceCount>0)
		{
			cudaSetDevice(localRank % deviceCount);
			cudaDeviceReset();
			cudaCheckLastError();
		}
	}
	////////////////////////////////////////////////////////////////////////////
	int size, rank;
	char name[MPI_MAX_PROCESSOR_NAME];
	int length;
		
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	
  	printf("Hello World from rank %02d, size %02d, of %s\n", 
		rank, size, name);
	////////////////////////////////////////////////////////////////////////////	
	int master 		= size-1;
	int worker;
	int numMasters 	= 1;
	int numWorkers 	= size-1;
	////////////////////////////////////////////////////////////////////////////
	int head = 0;
	int tail = size-2;
	////////////////////////////////////////////////////////////////////////////	
	// int deviceCount = 0;
	// if(rank != master)
	// {
		// cudaGetDeviceCount(&deviceCount);
		// cudaCheckLastError();
		// printf("There are %d device(s) at %s\n", deviceCount, name);
		// cudaSetDevice(rank % deviceCount);
		// cudaCheckLastError();
	// }
	////////////////////////////////////////////////////////////////////////////
	const int elemCount = 1024 * 1024 * 16; //1  GB
	float *h_src, *d_src, *h_dst, *d_dst;
	if(rank != master)
	{
		h_src = new float[elemCount];
		h_dst = new float[elemCount];
		cudaMalloc((void**)&d_src, elemCount*sizeof(float));
		cudaMalloc((void**)&d_dst, elemCount*sizeof(float));
	}
	////////////////////////////////////////////////////////////////////////////	
	double start, elapsed;
	MPI_Status stat;
	for(int k=1; k<16; k++)
	{
		////////////////////////////////////////////////////////////////////////////	
		MPI_Barrier(MPI_COMM_WORLD);
		start = MPI_Wtime();
		
		for (int i=0; i<100; i++)
		{
			if(rank == 0)
			{
				MPI_Send(d_src, elemCount, MPI_FLOAT, rank+k, 0, MPI_COMM_WORLD);
			}	
			if(rank == k)
			{
				MPI_Recv(d_dst, elemCount, MPI_FLOAT, rank-k, 0, MPI_COMM_WORLD, &stat);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		elapsed = MPI_Wtime() - start;
		////////////////////////////////////////////////////////////////////////////	
		if(rank == master)
		{
			printf("Bandwidth is: %.2fGB/s\n", 
			   (1.0f / (elapsed)) * ((100.0f * elemCount*sizeof(float)) / 1024.0f / 1024.0f / 1024.0f));
		}
	}
  	MPI_Finalize();
}