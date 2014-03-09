#include <mpi.h>
#include <iostream>
#include <cuda.h>
using namespace std;

// ----------------------------------------------------------------------------

#define checkLastError() {                                          				\
	cudaError_t error = cudaGetLastError();                               			\
	int id; 																		\
	cudaGetDevice(&id);																\
	if(error != cudaSuccess) {                                         				\
		printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",	\
			__FILE__,__LINE__, cudaGetErrorString(error), id);			      	 	\
		exit(EXIT_FAILURE);  														\
	}                                                               				\
}
// ----------------------------------------------------------------------------
void eikonal(float *solution, float *speedmap, int dimx, int dimy, int dimz);
// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
	/// Print the process information
	int size, rank;
	char name[MPI_MAX_PROCESSOR_NAME];
	int length;
		
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	
  	printf("Hello World from rank %02d, size %02d, of %s\n", rank, size, name);

	// Specify dimensions
	const int dimx  = 100;
	const int dimy  = 100;
	const int dimz  = 100;

	const int total = dimx*dimy*dimz;
	
	// Allocate host memory
	float *h_solution = new float[total];
	float *h_speedmap = new float[total];
	
	// Allocate device memory
	float *d_solution;
	float *d_speedmap;
	
	cudaMalloc((void**)&d_solution, total*sizeof(float));		checkLastError();
	cudaMalloc((void**)&d_speedmap, total*sizeof(float));		checkLastError();
	
	// Initialize the speedmap
	for(int z=0; z<dimz; z++)
	{
		for(int y=0; y<dimy; y++)
		{
			for(int x=0; x<dimx; x++)
			{
				h_speedmap[z*dimx*dimy+y*dimx+x] = 1.0f;
			}
		}
	}
	
	// Initialize the solution
	for(int z=0; z<dimz; z++)
	{
		for(int y=0; y<dimy; y++)
		{
			for(int x=0; x<dimx; x++)
			{
				h_solution[z*dimx*dimy+y*dimx+x] = 1000000.0f;
			}
		}
	}
	
	
	// Transferring to the device memory
	cudaMemcpy(d_solution, h_solution, total*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_speedmap, h_speedmap, total*sizeof(float), cudaMemcpyHostToDevice);
	
	// Invoking device kernel
	cudaMemcpy(h_solution, d_solution, total*sizeof(float), cudaMemcpyDeviceToHost);
	
  	MPI_Finalize();
	return 0;
}
// ----------------------------------------------------------------------------
__global__
void __eikonal(float *solution, float *speedmap, int dimx, int dimy, int dimz)
{
	int 	shared_index_1d 	= 	threadIdx.z * blockDim.y * blockDim.x + 
									threadIdx.y * blockDim.x + 
									threadIdx.x;
	int3 	shared_index_3d		= 	make_int3(shared_index_1d % (blockDim.y * blockDim.x),
											  shared_index_1d / (blockDim.y * blockDim.x),
											  shared_index_1d / blockDim.y / blockDim.x);
	int3	global_index_3d		=   make_int3(blockIdx.x * blockDim.x + shared_index_3d.x,
											  blockIdx.y * blockDim.y + shared_index_3d.y,
											  blockIdx.z * blockDim.z + shared_index_3d.z);
	int 	global_index_1d		=	global_index_3d.z * dimy * dimx +
									global_index_3d.y * dimx +
									global_index_3d.x;
	
}
void eikonal(float *solution, float *speedmap, int dimx, int dimy, int dimz)
{
	dim3 numBlocks((dimx/8 + ((dimx%8)?1:0)),
				   (dimy/8 + ((dimy%8)?1:0)),
				   (dimz/8 + ((dimz%8)?1:0)) );
	dim3 numThreads(8, 8, 8);
	__eikonal<<<numBlocks, numThreads>>>(solution, speedmap, dimx, dimy, dimz);
}
