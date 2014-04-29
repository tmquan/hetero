// #include <mpi.h>
// #include <iostream>

// using namespace std;

// int main(int argc, char** argv)
// {
	// int size, clusterRank;
	// char name[MPI_MAX_PROCESSOR_NAME];
	// int length;
		
	// MPI_Init(&argc, &argv);

	// MPI_Comm_rank(MPI_COMM_WORLD, &clusterRank);	
	// MPI_Comm_size(MPI_COMM_WORLD, &size);
	// MPI_Get_processor_name(name, &length);
	
  	// printf("Hello World from rank%02d, size %02d, of %s\n", clusterRank, size, name);

  	// MPI_Finalize();
	// return 0;
// }

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string>
#include <stdlib.h>		// For malloc 2d
// #include <opencv2/opencv.hpp>

#include <omp.h>
#include <mpi.h>
#include <cuda.h>
#include <helper_math.h> //For clamp

#include <assert.h>
#include <hetero_cmdparser.hpp>


using namespace std;
// using namespace cv;
////////////////////////////////////////////////////////////////////////////////////////////////////
#define cudaCheckLastError() {                                          			\
	cudaError_t error = cudaGetLastError();                               			\
	int id; cudaGetDevice(&id);                                                     \
	if(error != cudaSuccess) {                                                      \
		printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",	\
			__FILE__,__LINE__, cudaGetErrorString(error), id);                      \
		exit(EXIT_FAILURE);                                                         \
	}                                                                               \
}
////////////////////////////////////////////////////////////////////////////////////////////////////

#define checkReadFile(filename, pData, size) {                    					\
		fstream *fs = new fstream;													\
		fs->open(filename.c_str(), ios::in|ios::binary);							\
		if (!fs->is_open())															\
		{																			\
			printf("Cannot open file '%s' in file '%s' at line %i\n",				\
			filename, __FILE__, __LINE__);											\
			return 1;																\
		}																			\
		fs->read(reinterpret_cast<char*>(pData), size);								\
		fs->close();																\
		delete fs;																	\
	}																			
////////////////////////////////////////////////////////////////////////////////////////////////////
#define checkWriteFile(filename, pData, size) {                    					\
		fstream *fs = new fstream;													\
		fs->open(filename, ios::out|ios::binary);									\
		if (!fs->is_open())															\
		{																			\
			fprintf(stderr, "Cannot open file '%s' in file '%s' at line %i\n",		\
			filename, __FILE__, __LINE__);											\
			return 1;																\
		}																			\
		fs->write(reinterpret_cast<char*>(pData), size);							\
		fs->close();																\
		delete fs;																	\
	}
////////////////////////////////////////////////////////////////////////////////////////////////////
static void handle_error(int errcode, const char *str)
{
        char msg[MPI_MAX_ERROR_STRING];
        int resultlen;
        MPI_Error_string(errcode, msg, &resultlen);
        fprintf(stderr, "%s: %s\n", str, msg);
        MPI_Abort(MPI_COMM_WORLD, 1);
}
////////////////////////////////////////////////////////////////////////////////////////////////////	
const char* key =
	"{ h   |help        |      | print help message }"	
	"{ vx  |virtualDimx |      | virtualDimx }"
	"{ vy  |virtualDimy |      | virtualDimy }"
	"{ mp  |maxProcs    |  0   | maxProcs }"
	"{ id  |execId      |      | indicate the ith times launch mpi, can be considered as stride }"
	// "{ i   |srcFile   |      | source of the file }"
	// "{ dimx|dimx      |      | dimensionx }"
	// "{ dimy|dimy      |      | dimensiony }"
	// "{ cx  |clusterDimx      |      | clusterDimx }"
	// "{ cy  |clusterDimy      |      | clusterDimy }"
	;
////////////////////////////////////////////////////////////////////////////////////////////////////

#define at(x, y, dimx, dimy) ( clamp((int)y, 0, dimy-1)*dimx +     \
                               clamp((int)x, 0, dimx-1) )            

int main(int argc, char *argv[])
{
	//================================================================================
	// Initialize MPI
	int  rank, size;
	char name[MPI_MAX_PROCESSOR_NAME];
	int  length;
	int errCode;
	MPI_File fh;
	MPI_Init(&argc, &argv);
	//================================================================================
	// Retrieve the number of execId
	// Parsing the arguments
	CommandLineParser cmd(argc, argv, key);
	const int execId  			= cmd.get<int>("execId", false);
	const int maxProcs  		= cmd.get<int>("maxProcs", false);
	// const int clusterDimx  		= cmd.get<int>("clusterDimx", false);
	// const int clusterDimy  		= cmd.get<int>("clusterDimy", false);
	const int virtualDimx  		= cmd.get<int>("virtualDimx", false);
	const int virtualDimy  		= cmd.get<int>("virtualDimy", false);
	const int virtualSize  		= cmd.get<int>("virtualSize", false);

	// printf("execId=%d, setNumProcs=%d, rank=%d\n", execId, maxProcs, rank);		
	// printf("virtualDimx=%02d, virtualDimy=%02d\n", 
		    // virtualDimx, virtualDimy);
	//================================================================================
	// MPI_Comm comm2d;          		/* Cartesian communicator */
    // int dims[2];			   			/* allow MPI to choose grid block dimensions */
    // int periodic[2];			  		/* domain is non-periodic */
    // int reorder;		          		/* allow processes to be re-ranked */
    // int coords[2];            		/* coordinates of our block in grid */
    // int up, down;             		/* ranks of processes above and below ours */
    // int left, right;          		/* ranks of processes to each side of ours */
	
	///!!!
	// dims[0] = virtualDimx;	//clusterDimy;	//1;
	// dims[1] = virtualDimy;	//clusterDimx;	//2;
	// periodic[0] = 0;
	// periodic[1] = 0;
	// reorder = 1;
	
	// Set up Cartesian grid of processors. 
	// MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, reorder, &comm2d );
    // MPI_Cart_get(comm2d, 2, dims, periodic, coords );
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	MPI_Barrier(MPI_COMM_WORLD);
	//================================================================================
	// Manually determine rank in 2d, comm2d will work in this case
	int virtualRank = execId * maxProcs + rank;
	int2 virtualIdx = make_int2(virtualRank % virtualDimx,
								virtualRank / virtualDimx);
	printf("execId (%d), maxProcs(%d), rank (%d)\n", execId, maxProcs, rank);				
	printf("virtualIdx.x=%02d, virtualIdx.y=%02d, virtualRank=%02d, at %s\n", 
		    virtualIdx.x, virtualIdx.y, virtualRank, name);
	MPI_Barrier(MPI_COMM_WORLD);
	
	//================================================================================
	// Close MPI
	MPI_Finalize();
	return 0;
}