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
	"{ px  |processDimx |      | processDimx }"
	"{ py  |processDimy |      | processDimy }"
	"{ hx  |haloDimx    |      | haloDimx }"
	"{ hy  |haloDimy    |      | haloDimy }"
	"{ dimx|dimx        |      | dimensionx }"
	"{ dimy|dimy        |      | dimensiony }"
	"{ mp  |maxProcs    |  1   | maxProcs }"
	"{ id  |execId      |      | indicate the ith times launch mpi, act like a queue}"
	"{ i   |srcFile     |      | source of the file }"
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
	const int virtualDimx  		= cmd.get<int>("virtualDimx", false);	
	const int virtualDimy  		= cmd.get<int>("virtualDimy", false);
	
	const int processDimx  		= cmd.get<int>("processDimx", false);
	const int processDimy  		= cmd.get<int>("processDimy", false);

	const int haloDimx  		= cmd.get<int>("haloDimx", false);	
	const int haloDimy  		= cmd.get<int>("haloDimy", false);
	
	// const int virtualSize  		= cmd.get<int>("virtualSize", false);
	const int dimx  			= cmd.get<int>("dimx", false);
	const int dimy  			= cmd.get<int>("dimy", false);
	const string srcFile		= cmd.get<string>("srcFile", false);
	
	// printf("execId=%d, setNumProcs=%d, rank=%d\n", execId, maxProcs, rank);		
	// printf("virtualDimx=%02d, virtualDimy=%02d\n", 
		    // virtualDimx, virtualDimy);
	//================================================================================	
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
	printf("execId(%d), maxProcs(%d), rank(%d)\n", execId, maxProcs, rank);				
	MPI_Barrier(MPI_COMM_WORLD);
	printf("virtualIdx.x=%02d, virtualIdx.y=%02d, virtualRank=%02d, at %s\n", 
		    virtualIdx.x, virtualIdx.y, virtualRank, name);
	MPI_Barrier(MPI_COMM_WORLD);
	
	//================================================================================
	/// Data type primitives
	int starts[2];
	int subsizes[2];
	int bigsizes[2];
	MPI_Request request;
	
	//================================================================================
	///!!!Do not need to pack, calculate per-process directly
	/// !! First step: Determine size of buffer		
	int3 featureIdx		{  0,   0,	0};
	int3 processDim		{1, 1,  1};
	int2 index_2d;
	int3 closedChunkDim		{0, 0, 0};
	int3 halo {0, 0, 0};
	//================================================================================
	processDim.x = processDimx;
	processDim.y = processDimy;
	
	halo.x = haloDimx;
	halo.y = haloDimy;	
	//================================================================================
	for(featureIdx.y=0; featureIdx.y<processDim.y; featureIdx.y++)
	{
		int2 index_2d;
		for(featureIdx.x=0; featureIdx.x<processDim.x; featureIdx.x++)
		{
			//2D global index
			index_2d = make_int2(
				virtualIdx.x*processDim.x+featureIdx.x,
				virtualIdx.y*processDim.y+featureIdx.y);	
			if(index_2d.x==dimx) break;
		}
		if(index_2d.y==dimy) break;
	}				
	
	closedChunkDim = make_int3(featureIdx.x, featureIdx.y, 1);
	
	printf("Sub closed chunk size: closedChunkDim.x=%05d, closedChunkDim.y=%05d at virtualIdx.x=%02d, virtualIdx.y=%02d (virtualRank=%02d)\n", 
		closedChunkDim.x, closedChunkDim.y,
	    virtualIdx.x, virtualIdx.y, virtualRank, name);
	MPI_Barrier(MPI_COMM_WORLD);
	//================================================================================
	// Read the file
	char *ch = strdup(srcFile.c_str());
	cout << ch << endl;
	
	MPI_Datatype etype;
	etype = MPI_FLOAT;
	//================================================================================
	index_2d = make_int2(
		(virtualRank%virtualDimx)*processDim.x+0,
		(virtualRank/virtualDimx)*processDim.y+0);	
	
	
	///!Order is very important
	bigsizes[0] = dimy;
	bigsizes[1] = dimx;
	subsizes[0] = closedChunkDim.y;
	subsizes[1] = closedChunkDim.x;
	starts[0] 	= index_2d.y;
	starts[1] 	= index_2d.x;
	MPI_Datatype closedChunkArray;		///!!! Declare the data type
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &closedChunkArray);
	MPI_Type_commit(&closedChunkArray);	///!!! Commit the data type
	
	// Check correctness here
	/*
	errCode = MPI_File_open(MPI_COMM_WORLD, ch,	MPI_MODE_RDONLY,  MPI_INFO_NULL, &fh);
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	
	MPI_File_set_view(fh, 0, etype, closedChunkArray, "native", MPI_INFO_NULL);
	
	float *p_closedChunk;
	p_closedChunk = (float*)malloc(closedChunkDim.x*closedChunkDim.y*sizeof(float));
	
	MPI_File_read(fh, p_closedChunk, closedChunkDim.x*closedChunkDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	MPI_File_close(&fh);
	//================================================================================
	///!!! Write globally
	errCode = MPI_File_open(MPI_COMM_WORLD, "closedSubArray.raw",	MPI_MODE_RDWR|MPI_MODE_CREATE,  MPI_INFO_NULL, &fh);
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	MPI_File_set_view(fh, 0, etype, closedChunkArray, "native", MPI_INFO_NULL);
	MPI_File_write_all(fh, p_closedChunk, closedChunkDim.x*closedChunkDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 	
	MPI_File_close(&fh);
	*/
	//================================================================================
	//================================================================================
	//================================================================================
	//================================================================================
	int leftRank, rightRank;
	int topRank, bottomRank;
	
	leftRank 	= virtualIdx.x-1;
	rightRank 	= virtualIdx.x+1;
	topRank 	= virtualIdx.y-1;
	bottomRank 	= virtualIdx.y+1;
	
	///!!! Handle the boundary case
	bool atBoundariesLeftRight = (leftRank<0)|(rightRank>(virtualDimx-1));
	bool atBoundariesTopBottom = (topRank<0)|(bottomRank>(virtualDimy-1));
	
	int3 openedChunkDim		{0, 0, 0};
	openedChunkDim = make_int3(closedChunkDim.x + ((atBoundariesLeftRight)?(1*halo.x):(2*halo.x)),
							   closedChunkDim.y + ((atBoundariesTopBottom)?(1*halo.y):(2*halo.y)),
							   1);
	printf("Sub opened chunk size: openedChunkDim.x=%05d, openedChunkDim.y=%05d at virtualIdx.x=%02d, virtualIdx.y=%02d (virtualRank=%02d)\n", 
		openedChunkDim.x, openedChunkDim.y,
	    virtualIdx.x, virtualIdx.y, virtualRank, name);
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Redefine the data type
	bigsizes[0] = dimy;
	bigsizes[1] = dimx;
	subsizes[0] = openedChunkDim.y;
	subsizes[1] = openedChunkDim.x;
	starts[0] 	= (topRank<0)?0:(index_2d.y-halo.y);		///!!! Handle the boundary start indices
	starts[1] 	= (leftRank<0)?0:(index_2d.x-halo.x);		///!!! Handle the boundary start indices	
	MPI_Datatype openedChunkArray;		///!!! Declare the data type
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &openedChunkArray);
	MPI_Type_commit(&openedChunkArray);	///!!! Commit the data type
	
	errCode = MPI_File_open(MPI_COMM_WORLD, ch,	MPI_MODE_RDONLY,  MPI_INFO_NULL, &fh);
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	
	MPI_File_set_view(fh, 0, etype, openedChunkArray, "native", MPI_INFO_NULL);
	
	float *p_openedChunk;
	p_openedChunk = (float*)malloc(openedChunkDim.x*openedChunkDim.y*sizeof(float));
	
	MPI_File_read(fh, p_openedChunk, openedChunkDim.x*openedChunkDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	MPI_File_close(&fh);
	//================================================================================
	///!!! Processing with CUDA here
	/// Allocate d_src, d_dst, copy back to p_openedChunk
	//================================================================================
	// int3 closedChunkDim		{0, 0, 0};
	// closedChunkDim = make_int3(closedChunkDim.x,
							   // closedChunkDim.y,
							   // 1);
	// printf("Sub closed chunk size: closedChunkDim.x=%05d, closedChunkDim.y=%05d at virtualIdx.x=%02d, virtualIdx.y=%02d (virtualRank=%02d)\n", 
		// closedChunkDim.x, closedChunkDim.y,
	    // virtualIdx.x, virtualIdx.y, virtualRank, name);
		
	float *p_closedChunk;
	p_closedChunk = (float*)malloc(closedChunkDim.x*closedChunkDim.y*sizeof(float));	
	
	///!!! Act like shared memory, copy from read	
	bigsizes[0]  = openedChunkDim.y;
	bigsizes[1]  = openedChunkDim.x;
	subsizes[0]  = closedChunkDim.y;
	subsizes[1]  = closedChunkDim.x;
	starts[0] 	 = (topRank<0)?0:halo.y;
	starts[1] 	 = (leftRank<0)?0:halo.x;
	MPI_Datatype subarray;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
	MPI_Type_commit(&subarray);	///!!! Commit the data type
	//Self copy
	MPI_Isend(p_openedChunk, 1, subarray, rank, 0, MPI_COMM_WORLD, &request);	
	MPI_Recv(p_closedChunk, closedChunkDim.x*closedChunkDim.y , MPI_FLOAT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	
	///!Order is very important
	// bigsizes[0] = dimy;
	// bigsizes[1] = dimx;
	// subsizes[0] = closedChunkDim.y;
	// subsizes[1] = closedChunkDim.x;
	// starts[0] 	= index_2d.y;
	// starts[1] 	= index_2d.x;
	// MPI_Datatype closedChunkArray;		///!!! Declare the data type
	// MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
        // MPI_ORDER_C, MPI_FLOAT, &closedChunkArray);
	
	// MPI_Type_commit(&closedChunkArray);	///!!! Commit the data type
	
	///!!! Write globally
	errCode = MPI_File_open(MPI_COMM_WORLD, "processedSubArray.raw",	MPI_MODE_RDWR|MPI_MODE_CREATE,  MPI_INFO_NULL, &fh);
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	MPI_File_set_view(fh, 0, etype, closedChunkArray, "native", MPI_INFO_NULL);
	MPI_File_write_all(fh, p_closedChunk, closedChunkDim.x*closedChunkDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 	
	MPI_File_close(&fh);
	//================================================================================
	// Close MPI
	MPI_Finalize();
	return 0;
}