#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <mpi.h>
#include <helper_math.h>
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
///////////////////////////////////////////////////////////////////////////////
// Neumann Boundary Condition
#define at(x, y, z, dimx, dimy, dimz) (clamp(z, 0, dimz-1)*dimy*dimx		\
									  +clamp(y, 0, dimy-1)*dimx				\
									  +clamp(x, 0, dimx-1))	 
// ---------------------------------------------------------------------------- 
__global__
void __heatflow(float *src, float *dst, int dimx, int dimy, int dimz)
{
	int  index_1d;
	int3 index_3d;
	index_3d.x	=	blockIdx.x * blockDim.x + threadIdx.x;
	index_3d.y	=	blockIdx.y * blockDim.y + threadIdx.y;
	index_3d.z	=	blockIdx.z * blockDim.z + threadIdx.z;
	
	index_1d 	= index_3d.z * dimy * dimx + 
				  index_3d.y * dimx + 
				  index_3d.x;
	
	
	
	// Store back
	if (index_3d.z < dimz && 
		index_3d.y < dimy && 
		index_3d.x < dimx)
		// dst[index_1d] = src[index_1d];
		dst[at(index_3d.x, index_3d.y, index_3d.z, dimx, dimy, dimz)] 
		= (src[at(index_3d.x+1, index_3d.y+0, index_3d.z+0, dimx, dimy, dimz)] +
		   src[at(index_3d.x-1, index_3d.y+0, index_3d.z+0, dimx, dimy, dimz)] +
		   
		   src[at(index_3d.x+0, index_3d.y+1, index_3d.z+0, dimx, dimy, dimz)] +
		   src[at(index_3d.x+0, index_3d.y-1, index_3d.z+0, dimx, dimy, dimz)] +
		   
		   src[at(index_3d.x+0, index_3d.y+0, index_3d.z+1, dimx, dimy, dimz)] +
		   src[at(index_3d.x+0, index_3d.y+0, index_3d.z-1, dimx, dimy, dimz)]) /6.0f;
}
void heatflow(float *src, float *dst, int dimx, int dimy, int dimz)
{
	dim3 numBlocks((dimx/8 + ((dimx%8)?1:0)),
				   (dimy/8 + ((dimy%8)?1:0)),
				   (dimz/8 + ((dimz%8)?1:0)));
	dim3 numThreads(8, 8, 8);
	__heatflow<<<numBlocks, numThreads>>>(src, dst, dimx, dimy, dimz);
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
	int size, rank;
	char name[MPI_MAX_PROCESSOR_NAME];
	int length;
		
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	
  	printf("Hello World from rank %02d, size %02d, of %s\n", rank, size, name);

  	
	srand(time(NULL)); // for random number generator
	// Specify dimensions
	const int dimx  = 100;
	const int dimy  = 100;
	const int dimz  = 100;

	const int total = dimx*dimy*dimz;
	
	///
	int worker;
	int numMasters = 1;
	int numWorkers = size-1;
	
	const int master = size-1;
	const int head = 0;
	const int tail = size-2;
	
	float *h_src;
	float *h_dst;
	//Print out the paramemters
	if(rank == master)
	{
		////////////////////////////////////////////////////////////////////////////
		printf("Master process is allocating file from %d out of %d\n", rank, size);
		/// Print out the parameters
		printf("Size: %dx%dx%d\n", dimx, dimy, dimz);
		printf("Number of GPUs: %d\n", numWorkers);
		///  Allocate data from file and store to memory
		h_src  = new float[total];
		
		// Initialize the image source
		for(int z=0; z<dimz; z++)
		{
			for(int y=0; y<dimy; y++)
			{
				for(int x=0; x<dimx; x++)
				{
					// h_src[z*dimy*dimx+y*dimx+x] = (float)rand();
					h_src[z*dimy*dimx+y*dimx+x] = (float)(z*dimy*dimx+y*dimx+x);
				}
			}
		}
		
		h_dst  = new float[total];
	}		
	MPI::COMM_WORLD.Barrier();
	
	// // Determine temporal resolution for each process
	// int halo = 6;
	// int tmpz = dimz/numWorkers;
	// if(rank != master)
	// {
		// if(numWorkers == 1)
		// {
			// tmpz  = mTems;
			// dTotal = dimx*dimy*dTems;
		// }
		// else
		// {
			// if(rank == head)
			// {
				// dTems  = hTems + mTems;
				// dTotal = dimx*dimy*dTems;
			// }
			// else if(rank == tail)
			// {
				// dTems  = mTems + hTems;
				// dTotal = dimx*dimy*dTems;
			// }
			// else
			// {
				// dTems  = hTems + mTems + hTems;
				// dTotal = dimx*dimy*dTems;
			// }
		// }
	// }
	// // Allocate host memory
	// float *h_src = new float[total];
	// float *h_dst = new float[total];
	
	// // Allocate device memory
	// float *d_src;
	// float *d_dst;
	
	// cudaMalloc((void**)&d_src, total*sizeof(float));		checkLastError();
	// cudaMalloc((void**)&d_dst, total*sizeof(float));		checkLastError();
	
	// // Initialize the image source
	// for(int z=0; z<dimz; z++)
	// {
		// for(int y=0; y<dimy; y++)
		// {
			// for(int x=0; x<dimx; x++)
			// {
				// // h_src[z*dimy*dimx+y*dimx+x] = (float)rand();
				// h_src[z*dimy*dimx+y*dimx+x] = (float)(z*dimy*dimx+y*dimx+x);
			// }
		// }
	// }
	// // Transferring to the device memory
	// cudaMemcpy(d_src, h_src, total*sizeof(float), cudaMemcpyHostToDevice); checkLastError();
	
	// heatflow(d_src, d_dst, dimx, dimy, dimz);

	// cudaMemcpy(h_dst, d_dst, total*sizeof(float), cudaMemcpyDeviceToHost); checkLastError();
// cleanup:
	// cudaFree(d_src);
	// cudaFree(d_dst);
	// free(h_src);
	// free(h_dst);
	
	
	MPI_Finalize();
	return 0;
	
	
}
