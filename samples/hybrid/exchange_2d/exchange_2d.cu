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

#define checkReadFile(filename, pData, size) {                    				\
		fstream *fs = new fstream;												\
		fs->open(filename.c_str(), ios::in|ios::binary);						\
		if (!fs->is_open())														\
		{																		\
			printf("Cannot open file '%s' in file '%s' at line %i\n",			\
			filename, __FILE__, __LINE__);										\
			return 1;															\
		}																		\
		fs->read(reinterpret_cast<char*>(pData), size);							\
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
	"{ h   |help      |      | print help message }"
	"{ i   |srcFile   |      | source of the file }"
	"{ dimx|dimx      |      | dimensionx }"
	"{ dimy|dimy      |      | dimensiony }"
	;
////////////////////////////////////////////////////////////////////////////////////////////////////

#define at(x, y, dimx, dimy) ( clamp((int)y, 0, dimy-1)*dimx +     \
                               clamp((int)x, 0, dimx-1) )            

int main(int argc, char *argv[])
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
	// Initialize MPI
	int  rank, size;
	char name[MPI_MAX_PROCESSOR_NAME];
	int  length;
	int errCode;
	MPI_File fh;
	MPI_Init(&argc, &argv);
	
	// cout << "Terminate at " << __FILE__ << " " << __LINE__ << endl;	
	// MPI_Finalize(); return 0;
	
	//================================================================================
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
	
	///!!!
	dims[0] = 4;
	dims[1] = 4;
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
	string srcFile  		= cmd.get<string>("srcFile", false);
	const int dimx  		= cmd.get<int>("dimx", false);
	const int dimy  		= cmd.get<int>("dimy", false);
	const int total 		= dimx*dimy;
	
	
	float *h_src 			= new float[total];
	
	// Read to pointer in master process
	if(rank==master)		
	{
		// cout << "Here" << endl;
		// checkReadFile(srcFile, h_src, total*sizeof(float)); 	
		// cout << "Here" << endl;
	}
	// int3 clusterDim 	= make_int3(dims[0], dims[1], 1);
	int  processIdx_1d 	= rank;
	int3 processIdx_2d 	= make_int3(coords[1], coords[0], 1);
	
	/// Data type primitives
	int starts[2];
	int subsizes[2];
	int bigsizes[2];

	/// Mimic Pack and Unpack MPI
	int dimz = 1;
	
	int3 featureIdx		{  0,   0,	0};
	int3 processIdx		{  0,   0,	0};
	int3 processDim		{128, 128,  1};
	int3 subDataDim		{0, 0, 0};
	int3 clusterDim    	{(dimx/processDim.x + ((dimx%processDim.x)?1:0)),
						 (dimy/processDim.y + ((dimy%processDim.y)?1:0)),
						 (dimz/processDim.z + ((dimz%processDim.z)?1:0))};
	int2 index_2d;
	cout << "Cluster Dimension: " << clusterDim.x << " "  << clusterDim.y << endl;					 
	// float *tmp = new float[processDim.x * processDim.y]; // Create process beyond the sub problem size
	// cudaHostRegister(h_src, processDim.x * processDim.y *sizeof(float), cudaHostRegisterPortable);
	MPI_Request request;
	

	// cout << "Terminate at " << __FILE__ << " " << __LINE__ << endl;	MPI_Finalize(); return 0;
	
	//================================================================================

	///!!!Do not need to pack, calculate per-process directly
	/// !! First step: Determine size of buffer		
	processIdx = processIdx_2d;
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
	
	
	cout << "Receive preamble read at " << rank << endl;
	cout << "Sub problem size: " << subDataDim.x << " "  << subDataDim.y << " at " << rank << " " << coords[1] << " " << coords[0] << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	
	cout << "Terminate at " << __FILE__ << " " << __LINE__ << endl;
	// MPI_Finalize(); return 0;
	
	
	float *p_src;
	p_src = (float*)malloc(subDataDim.x*subDataDim.y*sizeof(float));
	// cudaMalloc((void**)&p_src, (subDataDim.x*subDataDim.y)*sizeof(float));
	
	
	//---------------------------------------------------------------------------------
	char *ch = strdup(srcFile.c_str());

	cout << ch << endl;
	
	MPI_Offset disp;
	disp = sizeof(float)*rank*processDim.x*processDim.y; 
	// disp = sizeof(float)*rank*subDataDim.x*subDataDim.y; 
	MPI_Datatype etype;
	etype = MPI_FLOAT;
	
	index_2d = make_int2(
		(rank%4)*processDim.x+0,
		(rank/4)*processDim.y+0);	
	// index_2d = make_int2(
		// (rank%4)*subDataDim.x+0,
		// (rank/4)*subDataDim.y+0);
		
	cout << "Start read from " << rank << endl;
	// int bigsizes[2]  = {dimy, dimx}; ///!Order is very important
	bigsizes[0] = dimy;
	bigsizes[1] = dimx;
	
	subsizes[0] = subDataDim.y;
	subsizes[1] = subDataDim.x;
	
	// int subsizes[2]  = {subDataDim.y, subDataDim.x}; ///!Order is very important
	// int subsizes[2]  = {processDim.y, processDim.x}; ///!Order is very important
	// int starts[2] 	 = {index_2d.y, index_2d.x}; ///!Order is very important
	starts[0] 	 = index_2d.y;
	starts[1] 	 = index_2d.x;
	
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "Start indices \t" << index_2d.x << " \t" << index_2d.y << " \t at " << rank << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Check the location here
	cout << "bigsizes \t" << bigsizes[1] << " \t" << bigsizes[0] << " \t at " << rank << "("<<coords[1]<<", "<<coords[0] <<")"<<endl;
	cout << "subsizes \t" << subsizes[1] << " \t" << subsizes[0] << " \t at " << rank << "("<<coords[1]<<", "<<coords[0] <<")"<<endl;
	cout << "startsId \t" << starts[1] << " \t" << starts[0] << " \t at " << rank << "("<<coords[1]<<", "<<coords[0] <<")"<< endl;
	// MPI_Finalize(); return 0;
	
	//-------------------------------------------------------------------------------------
	
	
	
	
	MPI_Datatype subarray;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
        MPI_ORDER_C, MPI_FLOAT, &subarray);
	
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	MPI_Type_commit(&subarray);
	errCode = MPI_File_open(MPI_COMM_WORLD, ch,	MPI_MODE_RDONLY,  MPI_INFO_NULL, &fh);
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;		
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	
	// MPI_File_set_view(fh, disp, etype, subarray, "native", MPI_INFO_NULL);
	MPI_File_set_view(fh, 0, etype, subarray, "native", MPI_INFO_NULL);
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	
	MPI_File_read(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	// MPI_File_read_all(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); // Process spawn and fail
	// MPI_File_read_ordered(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	// MPI_Type_free(&subarray);
	MPI_File_close(&fh);
	// MPI_Barrier(MPI_COMM_WORLD);	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;		
	if(p_src[0] !=0)
		cout << "Caught " << endl;
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	
	//---------------------------------------------------------------------------
	
	/// Debug, write partially
	MPI_Barrier(MPI_COMM_WORLD);
	char *filename = new char[100];
	sprintf(filename, "result_%02d_%02d.raw", processIdx_2d.x, processIdx_2d.y);
	printf("%s\n", filename);
	// float *h_tmp;
	// h_tmp = (float*)malloc(subDataDim.x*subDataDim.y*sizeof(float));
	// cudaHostRegister(h_tmp, subDataDim.x*subDataDim.y *sizeof(float), cudaHostRegisterPortable);
	// cudaMemcpy(h_tmp, p_src, subDataDim.x*subDataDim.y*sizeof(float), cudaMemcpyDeviceToHost); cudaCheckLastError();
	// checkWriteFile(filename, h_tmp, subDataDim.x*subDataDim.y*sizeof(float));
	// checkWriteFile(filename, p_src, processDim.x*processDim.y*sizeof(float));
	checkWriteFile(filename, p_src, subDataDim.x*subDataDim.y*sizeof(float));
	// MPI_Finalize(); return 0;
	// MPI_Finalize(); return 0;
	
	//---------------------------------------------
	//================================================================================
	///!!! Extend the memory to form another buffer
	int3 extDataDim		{0, 0, 0};
	int3 halo = {12, 12, 12};
	// if(processIdx_2d.x !=0) halo.x = 5;
	// if(processIdx_2d.y !=0) halo.y = 5;
	int leftRank, rightRank;
	int topRank, bottomRank;
	MPI_Cart_shift(comm2d, 0, 1, &topRank, &bottomRank);
	MPI_Cart_shift(comm2d, 1, 1, &leftRank, &rightRank);
	
	
	halo.x = 12;
	halo.y = 12;
	// extDataDim = make_int3(subDataDim.x + 2*halo.x,
						   // subDataDim.y + 2*halo.y,
						   // 1);
	///!!! Handle the boundary case
	bool atBoundariesLeftRight = (leftRank<0)|(rightRank<0);
	bool atBoundariesTopBottom = (topRank<0)|(bottomRank<0);
	extDataDim = make_int3(subDataDim.x + ((atBoundariesLeftRight)?(1*halo.x):(2*halo.x)),
						   subDataDim.y + ((atBoundariesTopBottom)?(1*halo.x):(2*halo.x)),
							1);
	cout << "At rank " << rank << "Extend data dim :" << extDataDim.x << " " << extDataDim.y<< endl;
	// Form another buffer
	float *p_ext;
	p_ext = (float*)malloc(extDataDim.x*extDataDim.y*sizeof(float));
	memset(p_ext, 0, extDataDim.x*extDataDim.y*sizeof(float));
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	
	// Redefine the data type
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = subDataDim.y;
	subsizes[1]  = subDataDim.x;
	starts[0] 	 = halo.y;
	starts[1] 	 = halo.x;
	
	///!!! Handle the boundary start indices
	if(leftRank<0) starts[1] = 0;	//x
	if(topRank<0)  starts[0] = 0;	//y
	
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &subarray);
	MPI_Type_commit(&subarray);
	
	// Self copy
	MPI_Isend(p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &request);	
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;
	MPI_Recv(p_ext, 1, subarray, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	MPI_Barrier(MPI_COMM_WORLD);
	sprintf(filename, "extend_%02d_%02d.raw", processIdx_2d.x, processIdx_2d.y);
	printf("%s\n", filename);
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	// checkWriteFile(filename, t_src, extDataDim.x*extDataDim.y*sizeof(float));
	checkWriteFile(filename, p_ext, extDataDim.x*extDataDim.y*sizeof(float));
	// checkWriteFile(filename, p_ext, subDataDim.x*subDataDim.y*sizeof(float));
	//================================================================================
	// Now we have to send the valid halo region to the ghost region
		

	// Redefine the data type
	// rightHalo
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = extDataDim.y;
	subsizes[1]  = halo.x;
	starts[0] 	 = 0;
	starts[1] 	 = extDataDim.x - 2*halo.x;
	MPI_Datatype rightHalo;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &rightHalo);
	MPI_Type_commit(&rightHalo);
	
	// leftGhost
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = extDataDim.y;
	subsizes[1]  = halo.x;
	starts[0] 	 = 0;
	starts[1] 	 = 0;
	MPI_Datatype leftGhost;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &leftGhost);
	MPI_Type_commit(&leftGhost);
	

	MPI_Isend(p_ext, 1, rightHalo, rightRank, 0, comm2d, &request);
	MPI_Recv(p_ext, 1, leftGhost, leftRank, 0, comm2d, MPI_STATUS_IGNORE);
	
	MPI_Barrier(MPI_COMM_WORLD);
	sprintf(filename, "extend_left2right_%02d_%02d.raw", processIdx_2d.x, processIdx_2d.y);
	printf("%s\n", filename);
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	checkWriteFile(filename, p_ext, extDataDim.x*extDataDim.y*sizeof(float)); 
	
	//---------------------------
	// leftHalo
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = extDataDim.y;
	subsizes[1]  = halo.x;
	starts[0] 	 = 0;
	starts[1] 	 = halo.x;
	MPI_Datatype leftHalo;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &leftHalo);
	MPI_Type_commit(&leftHalo);
	
	// rightGhost
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = extDataDim.y;
	subsizes[1]  = halo.x;
	starts[0] 	 = 0;
	starts[1] 	 = extDataDim.x - halo.x;
	MPI_Datatype rightGhost;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &rightGhost);
	MPI_Type_commit(&rightGhost);
	
	
	MPI_Isend(p_ext, 1, leftHalo, leftRank, 0, comm2d, &request);
	MPI_Recv(p_ext, 1, rightGhost, rightRank, 0, comm2d, MPI_STATUS_IGNORE);
	
	MPI_Barrier(MPI_COMM_WORLD);
	sprintf(filename, "extend_right2left_%02d_%02d.raw", processIdx_2d.x, processIdx_2d.y);
	printf("%s\n", filename);
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	checkWriteFile(filename, p_ext, extDataDim.x*extDataDim.y*sizeof(float)); 
	
	//---------------------------
	// bottomHalo
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = halo.y;
	subsizes[1]  = extDataDim.x;
	starts[0] 	 = extDataDim.y -2*halo.y;
	starts[1] 	 = 0;
	MPI_Datatype bottomHalo;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &bottomHalo);
	MPI_Type_commit(&bottomHalo);
	
	// topGhost
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = halo.y;
	subsizes[1]  = extDataDim.x;
	starts[0] 	 = 0;
	starts[1] 	 = 0;
	MPI_Datatype topGhost;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &topGhost);
	MPI_Type_commit(&topGhost);
	
	MPI_Isend(p_ext, 1, bottomHalo, bottomRank, 0, comm2d, &request);
	MPI_Recv(p_ext, 1, topGhost, topRank, 0, comm2d, MPI_STATUS_IGNORE);
	
	MPI_Barrier(MPI_COMM_WORLD);
	sprintf(filename, "extend_top2bottom_%02d_%02d.raw", processIdx_2d.x, processIdx_2d.y);
	printf("%s\n", filename);
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	checkWriteFile(filename, p_ext, extDataDim.x*extDataDim.y*sizeof(float)); 
	
	//---------------------------
	// topHalo
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = halo.y;
	subsizes[1]  = extDataDim.x;
	starts[0] 	 = halo.y;
	starts[1] 	 = 0;
	MPI_Datatype topHalo;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &topHalo);
	MPI_Type_commit(&topHalo);
	
	// bottomGhost
	bigsizes[0]  = extDataDim.y;
	bigsizes[1]  = extDataDim.x;
	subsizes[0]  = halo.y;
	subsizes[1]  = extDataDim.x;
	starts[0] 	 = extDataDim.y - halo.y;
	starts[1] 	 = 0;
	MPI_Datatype bottomGhost;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,  MPI_ORDER_C, MPI_FLOAT, &bottomGhost);
	MPI_Type_commit(&bottomGhost);
	
	MPI_Isend(p_ext, 1, topHalo, topRank, 0, comm2d, &request);
	MPI_Recv(p_ext, 1, bottomGhost, bottomRank, 0, comm2d, MPI_STATUS_IGNORE);
	
	MPI_Barrier(MPI_COMM_WORLD);
	sprintf(filename, "extend_bottom2top_%02d_%02d.raw", processIdx_2d.x, processIdx_2d.y);
	printf("%s\n", filename);
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	checkWriteFile(filename, p_ext, extDataDim.x*extDataDim.y*sizeof(float)); 
	//================================================================================
	///!!! Write globally
	// MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
        // MPI_ORDER_C, MPI_FLOAT, &subarray);
	// MPI_Type_commit(&subarray);
	
	// Delete the file before using that
	MPI_Barrier(MPI_COMM_WORLD);	
	// if(rank == master)
	// {
		// errCode = MPI_File_delete("test.raw", MPI_INFO_NULL);
		// if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_delete");
	// }
	// MPI_Barrier(MPI_COMM_WORLD);	
	
	// errCode = MPI_File_open(MPI_COMM_WORLD, "test.raw",	MPI_MODE_RDWR|MPI_MODE_CREATE,  MPI_INFO_NULL, &fh);
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;		
	// if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	
	// // MPI_File_set_view(fh, disp, etype, subarray,
	// MPI_File_set_view(fh, 0, etype, subarray, "native", MPI_INFO_NULL);
	// // MPI_Type_free(&subarray);
	// // MPI_File_set_view(fh, disp, etype, subarray, "native", MPI_INFO_NULL);
	// cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	cout << "At rank " << rank << endl;
	// cout << "Sub problem size will be written: " << subDataDim.x << " "  << subDataDim.y << endl;
	
	// // MPI_File_write(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	// // MPI_File_write(fh, p_src, 1, subarray, MPI_STATUS_IGNORE); 
	// MPI_File_write_all(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	// // MPI_File_write_all(fh, p_src, 1, subarray, MPI_STATUS_IGNORE); 
	// // MPI_File_write_ordered(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	// MPI_File_close(&fh);
	
	// // check identical
	// // if(rank==0)
	// // {
		// // float *ref = new float[dimx*dimy];
		// // float *arr = new float[dimx*dimy];
		
		// // checkReadFile(srcFile, ref, dimx*dimy*sizeof(float));
		// // // char file[10];
		// // string file = "test.raw";
		// // checkReadFile(file, arr, dimx*dimy*sizeof(float));
		// // // for(int y=0; y<dimy; y++)
		// // // {
			// // // for(int x=0; x<dimx; x++)
			// // // {
				// // // if(
			// // // }
		// // // }
		// // for(int k=0; k<total; k++)
		// // {
			// // if(ref[k] != arr[k])
			// // {
				// // cout << "Do not match at " << k << endl;
				// // goto cleanup;
			// // }
		// // }
		// // cout << "Matched!!!" << endl; 
		// // cleanup:
		// // free(ref);
		// // free(arr);
	// // }
	
	
 	MPI_Finalize();
	return 0;
	
}
	

