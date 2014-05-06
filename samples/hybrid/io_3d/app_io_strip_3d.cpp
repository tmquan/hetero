#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string>
#include <stdlib.h>		// For malloc 2d
// #include <opencv2/opencv.hpp>
#include <hdf5.h>
#include <omp.h>
#include <mpi.h>
#include <cuda.h>
#include <helper_math.h> //For clamp

#include <assert.h>
#include <hetero_cmdparser.hpp>

float ReverseFloat( const float inFloat )
{
   float retVal;
   char *floatToConvert = ( char* ) & inFloat;
   char *returnFloat = ( char* ) & retVal;

   // swap the bytes into a temporary buffer
   returnFloat[0] = floatToConvert[3];
   returnFloat[1] = floatToConvert[2];
   returnFloat[2] = floatToConvert[1];
   returnFloat[3] = floatToConvert[0];

   return retVal;
}

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
	"{ vz  |virtualDimz |      | virtualDimz }"
	"{ px  |processDimx |      | processDimx }"
	"{ py  |processDimy |      | processDimy }"
	"{ py  |processDimz |      | processDimz }"
	"{ hx  |haloDimx    |      | haloDimx }"
	"{ hy  |haloDimy    |      | haloDimy }"
	"{ hz  |haloDimy    |      | haloDimz }"
	"{ dimx|dimx        |      | dimensionx }"
	"{ dimy|dimy        |      | dimensiony }"
	"{ dimz|dimy        |      | dimensionz }"
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
	const int virtualDimz  		= cmd.get<int>("virtualDimz", false);
	
	const int processDimx  		= cmd.get<int>("processDimx", false);
	const int processDimy  		= cmd.get<int>("processDimy", false);
	const int processDimz  		= cmd.get<int>("processDimz", false);

	const int haloDimx  		= cmd.get<int>("haloDimx", false);	
	const int haloDimy  		= cmd.get<int>("haloDimy", false);
	const int haloDimz  		= cmd.get<int>("haloDimz", false);
	
	// const int virtualSize  		= cmd.get<int>("virtualSize", false);
	const int dimx  			= cmd.get<int>("dimx", false);
	const int dimy  			= cmd.get<int>("dimy", false);
	const int dimz  			= cmd.get<int>("dimz", false);
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
	// Manually determine rank in 3d, comm3d will work in this case
	int virtualRank = execId * maxProcs + rank;
	int3 virtualIdx = make_int3(virtualRank % (virtualDimx*virtualDimy) % virtualDimx,
								virtualRank % (virtualDimx*virtualDimy) / virtualDimx,
								virtualRank / (virtualDimx*virtualDimy));
	printf("execId(%d), maxProcs(%d), rank(%d)\n", execId, maxProcs, rank);				
	MPI_Barrier(MPI_COMM_WORLD);
	printf("virtualIdx.x=%02d, virtualIdx.y=%02d, virtualIdx.z=%02d, virtualRank=%02d, at %s\n", 
		    virtualIdx.x, virtualIdx.y, virtualIdx.z, virtualRank, name);
	MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Finalize();		return 0;
	//================================================================================
	/// Data type primitives
	int starts[3];
	int subsizes[3];
	int bigsizes[3];
	MPI_Request request;
	
	//================================================================================
	///!!!Do not need to pack, calculate per-process directly
	/// !! First step: Determine size of buffer		
	int3 featureIdx		{  0,   0,	0};
	int3 processDim		{1, 1,  1};
	int3 index_3d;
	int3 closedChunkDim		{0, 0, 0};
	int3 halo {0, 0, 0};
	//================================================================================
	processDim.x = processDimx;
	processDim.y = processDimy;
	processDim.z = processDimz;
	
	halo.x = haloDimx;
	halo.y = haloDimy;	
	halo.z = haloDimz;	
	//================================================================================
	for(featureIdx.z=0; featureIdx.z<processDim.z; featureIdx.z++)
	{
		int3 index_3d;
		for(featureIdx.y=0; featureIdx.y<processDim.y; featureIdx.y++)
		{
			
			for(featureIdx.x=0; featureIdx.x<processDim.x; featureIdx.x++)
			{
				//3D global index
				index_3d = make_int3(
					virtualIdx.x*processDim.x+featureIdx.x,
					virtualIdx.y*processDim.y+featureIdx.y,
					virtualIdx.z*processDim.z+featureIdx.z);	
				if(index_3d.x==dimx) break;
			}
			if(index_3d.y==dimy) break;
		}				
		if(index_3d.z==dimz) break;
	}
	closedChunkDim = make_int3(featureIdx.x, featureIdx.y, featureIdx.z);
	
	printf("Sub closed chunk size: closedChunkDim.x=%05d, closedChunkDim.y=%05d, closedChunkDim.z=%05d at virtualIdx.x=%02d, virtualIdx.y=%02d, virtualIdx.z=%02d (virtualRank=%02d)\n", 
		closedChunkDim.x, closedChunkDim.y, closedChunkDim.z,
	    virtualIdx.x, virtualIdx.y, virtualIdx.z, virtualRank, name);
	MPI_Barrier(MPI_COMM_WORLD);
	//================================================================================
	// Read the file
	char *ch = strdup(srcFile.c_str());
	cout << ch << endl;
	
	MPI_Datatype etype;
	etype = MPI_FLOAT;
	//================================================================================
	index_3d = make_int3(
		(virtualRank%(virtualDimx*virtualDimy)%virtualDimx)*processDim.x+0,
		(virtualRank%(virtualDimx*virtualDimy)/virtualDimx)*processDim.y+0,
		(virtualRank/(virtualDimx*virtualDimy)            )*processDim.z+0);	
	
	
	///!Order is very important
	bigsizes[0] = dimz; //0 2
	bigsizes[1] = dimy; //1 0
	bigsizes[2] = dimx;	//2 1
	subsizes[0] = closedChunkDim.z;
	subsizes[1] = closedChunkDim.y;
	subsizes[2] = closedChunkDim.x;
	starts[0] 	= index_3d.z;
	starts[1] 	= index_3d.y;
	starts[2] 	= index_3d.x;
	MPI_Datatype closedChunkArray;		///!!! Declare the data type
	MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &closedChunkArray);
	MPI_Type_commit(&closedChunkArray);	///!!! Commit the data type
	
	// Check correctness here
	/*
	errCode = MPI_File_open(MPI_COMM_WORLD, ch,	MPI_MODE_RDONLY,  MPI_INFO_NULL, &fh);
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	
	MPI_File_set_view(fh, 0, etype, closedChunkArray, "native", MPI_INFO_NULL);
	
	float *p_closedChunk;
	p_closedChunk = (float*)malloc(closedChunkDim.x*closedChunkDim.y*closedChunkDim.z*sizeof(float));
	
	MPI_File_read(fh, p_closedChunk, closedChunkDim.x*closedChunkDim.y*closedChunkDim.z, MPI_FLOAT, MPI_STATUS_IGNORE); 
	MPI_File_close(&fh);
	//================================================================================
	///!!! Write globally
	errCode = MPI_File_open(MPI_COMM_WORLD, "closedSubArray.raw",	MPI_MODE_RDWR|MPI_MODE_CREATE,  MPI_INFO_NULL, &fh);
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	MPI_File_set_view(fh, 0, etype, closedChunkArray, "native", MPI_INFO_NULL);
	MPI_File_write_all(fh, p_closedChunk, closedChunkDim.x*closedChunkDim.y*closedChunkDim.z, MPI_FLOAT, MPI_STATUS_IGNORE); 	
	MPI_File_close(&fh);
	*/
	// MPI_Finalize();		return 0;
	//================================================================================
	//================================================================================
	//================================================================================
	//================================================================================
	int leftRank, rightRank;
	int topRank, bottomRank;
	int frontRank, backRank;
	
	leftRank 	= virtualIdx.x-1;
	rightRank 	= virtualIdx.x+1;
	topRank 	= virtualIdx.y-1;
	bottomRank 	= virtualIdx.y+1;
	frontRank 	= virtualIdx.z-1;
	backRank 	= virtualIdx.z+1;
	
	///!!! Handle the boundary case
	bool atBoundariesLeftRight = (leftRank<0)|(rightRank>(virtualDimx-1));
	bool atBoundariesTopBottom = (topRank<0)|(bottomRank>(virtualDimy-1));
	bool atBoundariesFrontBack = (frontRank<0)|(backRank>(virtualDimz-1));
	
	int3 openedChunkDim		{0, 0, 0};
	openedChunkDim = make_int3(closedChunkDim.x + ((atBoundariesLeftRight)?((virtualDimx==1)?0*halo.x:1*halo.x):(2*halo.x)),
							   closedChunkDim.y + ((atBoundariesTopBottom)?((virtualDimy==1)?0*halo.y:1*halo.y):(2*halo.y)),
							   closedChunkDim.z + ((atBoundariesFrontBack)?((virtualDimz==1)?0*halo.z:1*halo.z):(2*halo.z)));
	printf("Sub opened chunk size: openedChunkDim.x=%05d, openedChunkDim.y=%05d, openedChunkDim.z=%05d at virtualIdx.x=%02d, virtualIdx.y=%02d, virtualIdx.z=%02d (virtualRank=%02d)\n", 
		openedChunkDim.x, openedChunkDim.y, openedChunkDim.z,
	    virtualIdx.x, virtualIdx.y, virtualIdx.z, virtualRank, name);
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Redefine the data type
	bigsizes[0] = dimz;
	bigsizes[1] = dimy;
	bigsizes[2] = dimx;
	subsizes[0] = openedChunkDim.z;
	subsizes[1] = openedChunkDim.y;
	subsizes[2] = openedChunkDim.x;
	starts[0] 	= (frontRank<0)?0:(index_3d.z-halo.z);		///!!! Handle the boundary start indices
	starts[1] 	= (topRank<0)?0:(index_3d.y-halo.y);		///!!! Handle the boundary start indices	
	starts[2] 	= (leftRank<0)?0:(index_3d.x-halo.x);		///!!! Handle the boundary start indices	
	MPI_Datatype openedChunkArray;		///!!! Declare the data type
	MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &openedChunkArray);
	MPI_Type_commit(&openedChunkArray);	///!!! Commit the data type
	
	errCode = MPI_File_open(MPI_COMM_WORLD, ch,	MPI_MODE_RDONLY,  MPI_INFO_NULL, &fh);
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	
	MPI_File_set_view(fh, 0, etype, openedChunkArray, "native", MPI_INFO_NULL);
	
	float *p_openedChunk;
	p_openedChunk = (float*)malloc(openedChunkDim.x*openedChunkDim.y*openedChunkDim.z*sizeof(float));
	
	MPI_File_read(fh, p_openedChunk, openedChunkDim.x*openedChunkDim.y*openedChunkDim.z, MPI_FLOAT, MPI_STATUS_IGNORE); 
	MPI_File_close(&fh);
	//================================================================================
	///!!! Processing with CUDA here
	/// Allocate d_src, d_dst, copy back to p_openedChunk
	float t;
	
	for(int k=0; k<openedChunkDim.x*openedChunkDim.y*openedChunkDim.z; k++)
	{
		// t = ReverseFloat(p_openedChunk[k]);
		// if(virtualRank==0)
			// p_openedChunk[k] = ReverseFloat(t/2);
		// else
			// p_openedChunk[k] = ReverseFloat(t);
		t = p_openedChunk[k];
		if(virtualRank==0)
			p_openedChunk[k] = t/2;
		else
			p_openedChunk[k] = t;
	}
	// float t = ReverseFloat(120.0f);
	// cout << t << " ";
	// memset(p_openedChunk, t, openedChunkDim.x*openedChunkDim.y*openedChunkDim.z*sizeof(float));
	//================================================================================
	// int3 closedChunkDim		{0, 0, 0};
	// closedChunkDim = make_int3(closedChunkDim.x,
							   // closedChunkDim.y,
							   // 1);
	// printf("Sub closed chunk size: closedChunkDim.x=%05d, closedChunkDim.y=%05d at virtualIdx.x=%02d, virtualIdx.y=%02d (virtualRank=%02d)\n", 
		// closedChunkDim.x, closedChunkDim.y,
	    // virtualIdx.x, virtualIdx.y, virtualRank, name);
		
	float *p_closedChunk;
	p_closedChunk = (float*)malloc(closedChunkDim.x*closedChunkDim.y*closedChunkDim.z*sizeof(float));	
	
	///!!! Act like shared memory, copy from read	
	bigsizes[0]  = openedChunkDim.z;
	bigsizes[1]  = openedChunkDim.y;
	bigsizes[2]  = openedChunkDim.x;
	subsizes[0]  = closedChunkDim.z;
	subsizes[1]  = closedChunkDim.y;
	subsizes[2]  = closedChunkDim.x;
	starts[0] 	 = (frontRank<0)?0:halo.z;
	starts[1] 	 = (topRank<0)?0:halo.y;
	starts[2] 	 = (leftRank<0)?0:halo.x;
	MPI_Datatype subarray;
	MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
	MPI_Type_commit(&subarray);	///!!! Commit the data type
	//Self copy
	MPI_Isend(p_openedChunk, 1, subarray, rank, 0, MPI_COMM_WORLD, &request);	
	MPI_Recv(p_closedChunk, closedChunkDim.x*closedChunkDim.y*closedChunkDim.z , MPI_FLOAT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	
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
	//================================================================================

	
	//================================================================================
	///!!! Write globally
	// errCode = MPI_File_open(MPI_COMM_WORLD, "processedSubArray.raw",	MPI_MODE_RDWR|MPI_MODE_CREATE,  MPI_INFO_NULL, &fh);
	MPI_Info info;
	MPI_Info_create(&info);
	/* no. of I/O devices to be used for file striping */
	MPI_Info_set(info, "striping_factor", "4");
	/* the striping unit in bytes */
	MPI_Info_set(info, "striping_unit", "900000"); //dimx*dimy*sizeof(float)
	errCode = MPI_File_open(MPI_COMM_WORLD, "processedSubArray.raw", 	MPI_MODE_RDWR|MPI_MODE_CREATE,	info, &fh);
	
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	// MPI_File_set_view(fh, 0, etype, closedChunkArray, "native", MPI_INFO_NULL);
	MPI_File_set_view(fh, 0, etype, closedChunkArray, "native", info);
	MPI_File_write_all(fh, p_closedChunk, closedChunkDim.x*closedChunkDim.y*closedChunkDim.z, MPI_FLOAT, MPI_STATUS_IGNORE); 	
	MPI_File_close(&fh);
	//================================================================================
	// Close MPI
	MPI_Info_free(&info);
	free(p_openedChunk);
	free(p_closedChunk);
	MPI_Finalize();
	return 0;
}