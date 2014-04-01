#include <iostream>
#include <iomanip>      // std::setfill, std::setw
#include <mpi.h>
#include <cuda.h>
#include <hetero_cmdparser.hpp>

using namespace std;

const char* key =
	"{ h   |help      |      | print help message }"
	"{ i   |srcFile   |      | source of the file }"
	;

int main(int argc, char *argv[])
{
	MPI_Comm comm2d;          		/* Cartesian communicator */
    int dims[2] = { 0, 0 };   		/* allow MPI to choose grid block dimensions */
    int periodic[2] = { 0, 0 };  	/* domain is non-periodic */
    int reorder = 1;          		/* allow processes to be re-ranked */
    int coords[2];            		/* coordinates of our block in grid */
    int up, down;             		/* ranks of processes above and below ours */
    int left, right;          		/* ranks of processes to each side of ours */
	
	// Initialize MPI
	int  rank, size;
	char name[MPI_MAX_PROCESSOR_NAME];
	int  length;
		
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	
  	printf("This is rank %02d, size %02d, of %s\n", rank, size, name);

	// Parsing the arguments
	CommandLineParser cmd(argc, argv, key);
	cmd.printParams();
	MPI_Barrier(MPI_COMM_WORLD);
	
    // Set up Cartesian grid of processors.  A new communicator is
    // created we get our rank within it. 
    MPI_Dims_create( rank, 2, dims );
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periodic, reorder, &comm2d );
    MPI_Cart_get( comm2d, 2, dims, periodic, coords );
    MPI_Comm_rank( comm2d, &rank );

    
    // Figure out who my neighbors are.  left, right, down, and up will
    // be set to the rank of the process responsible for the corresponding
    // block relative to the position of the block we are responsible for.
    // If there is no neighbor in a particular direction the returned rank
    // will be MPI_PROC_NULL which will be ignored by subsequent
    // MPI_sendrecv() calls.
    MPI_Cart_shift( comm2d, 0, 1, &left, &right);
    MPI_Cart_shift( comm2d, 1, 1, &up, 	 &down);
	
	// // Declare the necessary indices
	// int2 gridDim	{2, 2};
	// int2 blockDim	{4, 4};
	// int2 blockIdx	{0, 0};
	// int blockIdx_1d;
	// int halo = 1;
	
	// int3 blockDimWithHalo{blockDim.x + 2*halo,
                          // blockDim.y + 2*halo};
    // int blockSizeWithHalo    = (blockDim.x + 2*halo) * (blockDim.y + 2*halo); 
    // int blockSize            = (blockDim.x + 0*halo) * (blockDim.y + 0*halo);
	
	// int numBatches = (blockSizeWithHalo/blockSize + ((blockSizeWithHalo%blockSize)?1:0));
    // int batch = 0; //For reading below
	
	// blockIdx_1d = blockIdx.y*blockDim.y + 
				  // blockIdx.x;
	
	// // int2 threadDim	{4, 4};
	// int2 threadIdx	{0, 0};
	
	// int  index_1d;
	// int2 index_2d;
	// float *h_src = new float[8*8];
	// float *h_dst = new float[8*8];
	
	// int dimx = 8;
	// int dimy = 8;
	
	// int2 shared_index_2d{0, 0};
	// int  shared_index_1d = 0;
	
	// int2 global_index_2d{0, 0};
	// int  global_index_1d = 0;
	
	// // #pragma omp parallel 
	// for(blockIdx.y=0; blockIdx.y<gridDim.y; blockIdx.y++)
	// {
		// // #pragma omp parallel 
		// for(blockIdx.x=0; blockIdx.x<gridDim.x; blockIdx.x++)
		// {
			// if(rank == blockIdx_1d)
			// {
				// // #pragma omp parallel 
				// for(threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y++)
				// {
					// // // #pragma omp parallel 					
					// // for(threadIdx.x=0; threadIdx.x<blockDim.x; threadIdx.x++)
					// // {
						// // index_2d = make_int2(blockIdx.x*blockDim.x+threadIdx.x,
											 // // blockIdx.y*blockDim.y+threadIdx.y);
						// // h_src[index_2d.y*8+index_2d.x] = 1		* threadIdx.x+
														 // // 10		* threadIdx.y+
														 // // 100 	* blockIdx.x +
														 // // 1000	* blockIdx.y;
														 
						// // printf("\t%04.0f", h_src[index_2d.y*8+index_2d.x]);
						// // if (threadIdx.x==(blockDim.x-1))	printf("\n");	
					// // }
					
					// for(batch=0; batch<numBatches; batch++)
					// {
						// shared_index_1d 	= threadIdx.y * blockDim.x + 
											  // threadIdx.x +
											  // blockDim.x  * blockDim.y * batch;  // Next number of loading, magic is here
											  
						// shared_index_2d.x   = (shared_index_1d % (blockDimWithHalo.y * blockDimWithHalo.x)) % blockDimWithHalo.x;
						// shared_index_2d.y   = (shared_index_1d % (blockDimWithHalo.y * blockDimWithHalo.x)) / blockDimWithHalo.x;
						
						// global_index_2d.x   = blockIdx.x * blockDim.x + shared_index_2d.x - halo;
						// global_index_2d.y   = blockIdx.y * blockDim.y + shared_index_2d.y - halo;
						
						// global_index_1d 	= global_index_2d.y * dimx + 
											  // global_index_2d.x;
						// if (shared_index_2d.y < blockDimWithHalo.y) 
						// {
							// if (global_index_2d.y >= 0 && global_index_2d.y < dimy &&
								// global_index_2d.x >= 0 && global_index_2d.x < dimx )	
							// {
								// h_src[global_index_1d] = 1		* global_index_2d.x+
														// 10		* global_index_2d.y+
														// 100 	* blockIdx.x +
														// 1000	* blockIdx.y;
								// printf("\t%04.0f", h_src[global_index_1d]);
								// if (shared_index_2d.x==(blockDimWithHalo.x-1))	printf("\n");	
							// }
						// }
					// }
				// }
				// printf("=============================================================\n");
			// }
		// }
	// }
	// Generate 1D data
	// float *h_src = new float[8*8];
	// float *h_dst = new float[8*8];
	
	// int dimx = 8;
	// int dimy = 8;
	// cout << "Size of data : " << sizeof(h_data) << endl;	
	// Since this is a demonstration program, here we initialize the
	// grid with values that indicate their original position in the
	// grid.  Assuming NX and NY are 100 or smaller then these values
	// have the form R.XXYY where:
	// R  is the rank of of the process that created the data
	// XX is the x coordinate in the grid (0 is at left)
	// YY is the y coordinate in the grid (0 is at bottom)
     
    // for (int y=0; y<dimy; y++ )
    // {
        // for (int x=0; x<dimx; x++)
        // {
            // // [i * halo_grid.ny + j] = rank
                // // + 0.01   * ( i + halo_grid.x0 ) 
                // // + 0.0001 * ( j + halo_grid.y0 );
        // }
    // }
	
  	MPI_Finalize();
	return 0;
}