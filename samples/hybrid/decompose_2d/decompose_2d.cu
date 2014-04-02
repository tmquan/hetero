#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string>

#include <omp.h>
#include <mpi.h>
#include <cuda.h>

#include <assert.h>
#include <hetero_cmdparser.hpp>


using namespace std;
////////////////////////////////////////////////////////////////////////////////////////////////////

#define checkReadFile(filename, pData, size) {                    				\
		fstream *fs = new fstream;												\
		fs->open(filename.c_str(), ios::in|ios::binary);						\
		if (!fs->is_open())														\
		{																		\
			printf("Cannot open file '%s' in file '%s' at line %i\n",			\
			filename, __FILE__, __LINE__);										\
			goto cleanup;														\
		}																		\
		fs->read(reinterpret_cast<char*>(pData), size);							\
cleanup :																		\
		fs->close();															\
		delete fs;																\
	}																			
////////////////////////////////////////////////////////////////////////////////////////////////////
#define checkWriteFile(filename, pData, size) {                    				\
		fstream *fs = new fstream;												\
		fs->open(filename, ios::out|ios::binary);								\
		if (!fs->is_open())														\
		{																		\
			fprintf(stderr, "Cannot open file '%s' in file '%s' at line %i\n",	\
			filename, __FILE__, __LINE__);										\
			return 1;															\
		}																		\
		fs->write(reinterpret_cast<char*>(pData), size);						\
		fs->close();															\
		delete fs;																\
	}
////////////////////////////////////////////////////////////////////////////////////////////////////	
const char* key =
	"{ h   |help      |      | print help message }"
	"{ i   |srcFile   |      | source of the file }"
	"{ dimx|dimx      |      | dimensionx }"
	"{ dimy|dimy      |      | dimensiony }"
	;
////////////////////////////////////////////////////////////////////////////////////////////////////
#define grid_side 3
int main(int argc, char *argv[])
{
	
	// Initialize MPI
	int  rank, size;
	char name[MPI_MAX_PROCESSOR_NAME];
	int  length;
		
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	
  	printf("This is rank %02d, size %02d, of %s\n", rank, size, name);
	MPI_Barrier(MPI_COMM_WORLD);
	//
	MPI_Comm comm2d;          		/* Cartesian communicator */
    int dims[2];// = {0, 0};   		/* allow MPI to choose grid block dimensions */
    int periodic[2];// = {0, 0};  	/* domain is non-periodic */
    int reorder;// = 1;          	/* allow processes to be re-ranked */
    int coords[2];            		/* coordinates of our block in grid */
    int up, down;             		/* ranks of processes above and below ours */
    int left, right;          		/* ranks of processes to each side of ours */
	
	//
	int master 		= 0;
	int worker;
	int numMasters 	= 1;
	int numWorkers 	= size;
	
	// Parsing the arguments
	CommandLineParser cmd(argc, argv, key);
	if(rank==master)	cmd.printParams();
	MPI_Barrier(MPI_COMM_WORLD);
	
	dims[0] = 2;
	dims[1] = 2;
	periodic[0] = 0;
	periodic[1] = 0;
	reorder = 1;
	// // Set up Cartesian grid of processors.  A new communicator is
    // // created we get our rank within it. 
    // MPI_Dims_create(rank, 2, dims); ///This line will not work
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, reorder, &comm2d );
    MPI_Cart_get(comm2d, 2, dims, periodic, coords );
    MPI_Comm_rank(comm2d, &rank );
	printf("%d, %d, %d\n",coords[0],coords[1],rank);

	MPI_Barrier(MPI_COMM_WORLD);
	
	// Retrieve the information from cmd
	const string srcFile  	= cmd.get<string>("srcFile", false);
	const int dimx  		= cmd.get<int>("dimx", false);
	const int dimy  		= cmd.get<int>("dimy", false);
	const int total 		= dimx*dimy;
	
	
	float *h_src 			= new float[total];
	
	// Read to pointer in master process
	if(rank==master)		
	{
		cout << "Here" << endl;
		checkReadFile(srcFile, h_src, total*sizeof(float)); 	
		cout << "Here" << endl;
	}
	// int3 clusterDim 	= make_int3(dims[0], dims[1], 1);
	int  processIdx_1d 	= rank;
	int3 processIdx_2d 	= make_int3(coords[0], coords[1], 1);
	
	

	/// Mimic Pack and Unpack MPI
	int dimz = 1;

	int3 featureIdx		{  0,   0,	0};
	int3 processIdx		{  0,   0,	0};
	int3 processDim		{256, 256, 1};
	int3 subDataDim		{0, 0, 0};
	int3 clusterDim    	{(dimx/processDim.x + ((dimx%processDim.x)?1:0)),
						 (dimy/processDim.y + ((dimy%processDim.y)?1:0)),
						 (dimz/processDim.z + ((dimz%processDim.z)?1:0))};
						 
	float *tmp = new float[processDim.x * processDim.y]; // Create process beyond the sub problem size
	MPI_Request request;
	
	float *p_src = (float*)malloc(processDim.x*processDim.y*sizeof(float));
	//Start packing
	/// Naive approach, copy to another buffer, then send
	int2 index_2d;
	double start = MPI_Wtime();
	int caught = 0;
	if(rank==master)
	{
		for(processIdx.y=0; processIdx.y<clusterDim.y; processIdx.y++)
		{
			for(processIdx.x=0; processIdx.x<clusterDim.x; processIdx.x++)
			{
				// First step: Determine size of buffer
				
				for(featureIdx.y=0; featureIdx.y<processDim.y; featureIdx.y++)
				{
					for(featureIdx.x=0; featureIdx.x<processDim.x; featureIdx.x++)
					{
						//2D global index
						index_2d = make_int2(
							processIdx.x*processDim.x+featureIdx.x,
							processIdx.y*processDim.y+featureIdx.y);	
						if(index_2d.x==dimx) break;
					}
					if(index_2d.y==dimy) break;
				}				
				subDataDim = make_int3(featureIdx.x, featureIdx.y, 1);
				cout << "Sub problem size: " << subDataDim.x << " "  << subDataDim.y << endl;
				// tmp = (float*)malloc(subDataDim.x*subDataDim.y * sizeof(float));
				// float *tmp;
				// tmp = (float*)realloc(tmp, subDataDim.x*subDataDim.y * sizeof(float));
				// tmp = (float*)realloc(tmp, subDataDim.x*subDataDim.y * sizeof(float));
				// tmp = new float[subDataDim.x*subDataDim.y];
				
				//Second step: copy subdataSize
				for(featureIdx.y=0; featureIdx.y<processDim.y; featureIdx.y++)
				{
					for(featureIdx.x=0; featureIdx.x<processDim.x; featureIdx.x++)
					{
						if(featureIdx.x == 0) // First position of first block
						{
							//2D global index
							index_2d = make_int2(
								processIdx.x*processDim.x+featureIdx.x,
								processIdx.y*processDim.y+featureIdx.y);		
							if(index_2d.y<dimy)
							{
								// cout << "Caught " << ++caught << endl;
								memcpy(
									// &tmp[featureIdx.y * processDim.x],
									&tmp[featureIdx.y * subDataDim.x],
									&h_src[index_2d.y*dimx + index_2d.x],
									// processDim.x*sizeof(float));
									subDataDim.x*sizeof(float));
									
								// std::swap(
									// tmp[featureIdx.y * processDim.x],
									// h_src[index_2d.y*dimx + index_2d.x]
									// );
							}
						}						
					}
				}
				

				processIdx_1d = processIdx.y * clusterDim.x + processIdx.x;
				cout << processIdx_1d << endl;
				
				// Send to worker process
				// MPI_Isend(tmp, processDim.x*processDim.y, MPI_FLOAT, processIdx_1d, 0, MPI_COMM_WORLD, &request);	
				// Send the size of message
				MPI_Isend(&subDataDim, 1, MPI_DOUBLE, processIdx_1d, 0, MPI_COMM_WORLD, &request);	
				// Send the message
				MPI_Isend(tmp, subDataDim.x *  subDataDim.y, MPI_FLOAT, processIdx_1d, 1, MPI_COMM_WORLD, &request);	
			
				cout << "Sent" << endl;
				// free(tmp);
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Recv(p_src, processDim.x*processDim.y, MPI_FLOAT, master, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&subDataDim, 1, MPI_DOUBLE, master, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(p_src, subDataDim.x *  subDataDim.y, MPI_FLOAT, master, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	double elapsed = MPI_Wtime() - start;
	if(rank==master) cout << "Time : " << elapsed << " s " << endl;
	/// Debug
	MPI_Barrier(MPI_COMM_WORLD);
	char *filename = new char[100];
	sprintf(filename, "result_%02d_%02d.raw", processIdx_2d.x, processIdx_2d.y);
	printf("%s\n", filename);
	checkWriteFile(filename, p_src, processDim.x*processDim.y*sizeof(float));
	
	
	
 	MPI_Finalize();
	return 0;
}
	