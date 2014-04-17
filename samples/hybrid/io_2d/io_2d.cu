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
#define grid_side 3
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
	string srcFile  	= cmd.get<string>("srcFile", false);
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
	int3 processIdx_2d 	= make_int3(coords[0], coords[1], 1);
	
	

	/// Mimic Pack and Unpack MPI
	int dimz = 1;

	int3 featureIdx		{  0,   0,	0};
	int3 processIdx		{  0,   0,	0};
	int3 processDim		{256, 256,  1};
	int3 subDataDim		{0, 0, 0};
	int3 clusterDim    	{(dimx/processDim.x + ((dimx%processDim.x)?1:0)),
						 (dimy/processDim.y + ((dimy%processDim.y)?1:0)),
						 (dimz/processDim.z + ((dimz%processDim.z)?1:0))};
	cout << "Cluster Dimension: " << clusterDim.x << " "  << clusterDim.y << endl;					 
	// float *tmp = new float[processDim.x * processDim.y]; // Create process beyond the sub problem size
	// cudaHostRegister(h_src, processDim.x * processDim.y *sizeof(float), cudaHostRegisterPortable);
	MPI_Request request;
	

	
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
				/// !!! First step: Determine size of buffer				
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
				
				//Second step: copy subdataSize
				index_2d = make_int2(
					processIdx.x*processDim.x+0,
					processIdx.y*processDim.y+0);	
				MPI_Datatype mysubarray;
				int starts[2] 	 = {index_2d.y, index_2d.x}; ///!Order is very important
				int subsizes[2]  = {subDataDim.y, subDataDim.x}; ///!Order is very important
				int bigsizes[2]  = {dimy, dimx}; ///!Order is very important
				MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
                                 MPI_ORDER_C, MPI_FLOAT, &mysubarray);
				MPI_Type_commit(&mysubarray);
				
				
				// for(featureIdx.y=0; featureIdx.y<processDim.y; featureIdx.y++)
				// {
					// for(featureIdx.x=0; featureIdx.x<processDim.x; featureIdx.x++)
					// {
						// if(featureIdx.x == 0) // First position of first block
						// {
							// //2D global index
							// index_2d = make_int2(
								// processIdx.x*processDim.x+featureIdx.x,
								// processIdx.y*processDim.y+featureIdx.y);		
							// if(index_2d.y<dimy)
							// {
								// // cout << "Caught " << ++caught << endl;
								// memcpy(
									// // &tmp[featureIdx.y * processDim.x],
									// &tmp[featureIdx.y * subDataDim.x],
									// &h_src[index_2d.y*dimx + index_2d.x],
									// // processDim.x*sizeof(float));
									// subDataDim.x*sizeof(float));
							// }
						// }						
					// }
				// }		

				processIdx_1d = processIdx.y * clusterDim.x + processIdx.x;
				cout << processIdx_1d << endl;
				
				/// !!! Send to worker process
				// Send the size of message
				MPI_Isend(&subDataDim, 1, MPI_DOUBLE, processIdx_1d, 0, MPI_COMM_WORLD, &request);	
				// Send the message
				// MPI_Isend(tmp, subDataDim.x *  subDataDim.y, MPI_FLOAT, processIdx_1d, 1, MPI_COMM_WORLD, &request);	
				// MPI_Isend(h_src, 1, mysubarray, processIdx_1d, 1, MPI_COMM_WORLD, &request);	
				// MPI_Send(&(bigarray[0][0]), 1, mysubarray, receiver, ourtag, MPI_COMM_WORLD);
				cout << "Sent" << endl;
				// free(tmp);
				
				
				MPI_Type_free(&mysubarray);
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Recv(p_src, processDim.x*processDim.y, MPI_FLOAT, master, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&subDataDim, 1, MPI_DOUBLE, master, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	//MPI_Recv(p_src, subDataDim.x *  subDataDim.y, MPI_FLOAT, master, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	cout << "Receive preamble read from " << rank << endl;
	cout << "Sub problem size: " << subDataDim.x << " "  << subDataDim.y << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	
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
		(rank%2)*processDim.x+0,
		(rank/2)*processDim.y+0);	
	// index_2d = make_int2(
		// (rank%2)*subDataDim.x+0,
		// (rank/2)*subDataDim.y+0);
		
	cout << "Start read from " << rank << endl;
	int bigsizes[2]  = {dimy, dimx}; ///!Order is very important
	int subsizes[2]  = {subDataDim.y, subDataDim.x}; ///!Order is very important
	// int subsizes[2]  = {processDim.y, processDim.x}; ///!Order is very important
	int starts[2] 	 = {index_2d.y, index_2d.x}; ///!Order is very important
	
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "Start indices \t" << index_2d.x << " \t" << index_2d.y << " \t at " << rank << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Datatype subarray;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
        MPI_ORDER_C, MPI_FLOAT, &subarray);
	
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
	MPI_Type_free(&subarray);
	MPI_File_close(&fh);
	// MPI_Barrier(MPI_COMM_WORLD);	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;		
	if(p_src[0] !=0)
		cout << "Caught " << endl;
	cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	//------------------------------------------------------------------------------
	// fstream fs;		
	// // if(rank==0)
	// // {
	// fs.open(srcFile.c_str(), ios::in|ios::binary);						
	// // cout << "Start read from " << rank << endl;
	// if (!fs.is_open())														
	// {																		
		// printf("Cannot open file '%s' in file '%s' at line %i\n",			
		// srcFile, __FILE__, __LINE__);										
		// return 1;															
	// }																		
																
	// // cout << "File opened from " << rank << endl;
	// // cout << "Sub problem size: " << subDataDim.x << " "  << subDataDim.y << endl;
	// // cout << "Dimension size: " << dimx << " "  << dimy << endl;
	
	// processIdx.x = rank%clusterDim.x;
	// processIdx.y = rank/clusterDim.x;
	
	// for(featureIdx.y=0; featureIdx.y<subDataDim.y; featureIdx.y++)
	// {
		// for(featureIdx.x=0; featureIdx.x<subDataDim.x; featureIdx.x++)
		// {
			// if(featureIdx.x == 0) // First position of row
			// {
				// //2D global index
				// index_2d = make_int2(
					// processIdx.x*subDataDim.x+featureIdx.x,
					// processIdx.y*subDataDim.y+featureIdx.y);	
				// // cout << "Global Index 2d: " << index_2d.x << " "  << index_2d.y << endl;
				// if(index_2d.y<dimy) //For handling the boundary problem
				// {					
					// fs.seekg((index_2d.y*dimx + index_2d.x)*sizeof(float), ios::beg);					
					// fs.read(reinterpret_cast<char*>(&p_src[featureIdx.y * subDataDim.x]), subDataDim.x*sizeof(float));
					// // if(p_src[featureIdx.y * subDataDim.x] !=0)
						// // cout << "Caught " << ++caught << endl;
				// }
			// }						
		// }
	// }	
	
	// fs.close();			
	//------------------------------------------------------------------------------
	
	
	
		
	MPI_Barrier(MPI_COMM_WORLD);	
	cout << "Finish read from " << rank << endl;
	double elapsed = MPI_Wtime() - start;
	if(rank==master) cout << "Time : " << elapsed << " s " << endl;
	
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
	
	
	///!!! Write globally
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
        MPI_ORDER_C, MPI_FLOAT, &subarray);
	
	MPI_Type_commit(&subarray);
	// errCode = MPI_File_delete("test.raw", MPI_INFO_NULL);
	// if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_delete");
	errCode = MPI_File_open(MPI_COMM_WORLD, "test.raw",	MPI_MODE_RDWR|MPI_MODE_CREATE,  MPI_INFO_NULL, &fh);
	// cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;		
	if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
	
	// MPI_File_set_view(fh, disp, etype, subarray,
	MPI_File_set_view(fh, 0, etype, subarray, "native", MPI_INFO_NULL);
	MPI_Type_free(&subarray);
	// MPI_File_set_view(fh, disp, etype, subarray, "native", MPI_INFO_NULL);
	// cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
	cout << "At rank " << rank << endl;
	cout << "Sub problem size will be written: " << subDataDim.x << " "  << subDataDim.y << endl;
	
	// MPI_File_write(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	// MPI_File_write(fh, p_src, 1, subarray, MPI_STATUS_IGNORE); 
	MPI_File_write_all(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	// MPI_File_write_all(fh, p_src, 1, subarray, MPI_STATUS_IGNORE); 
	// MPI_File_write_ordered(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
	MPI_File_close(&fh);
	
	// check identical
	// if(rank==0)
	// {
		// float *ref = new float[dimx*dimy];
		// float *arr = new float[dimx*dimy];
		
		// checkReadFile(srcFile, ref, dimx*dimy*sizeof(float));
		// // char file[10];
		// string file = "test.raw";
		// checkReadFile(file, arr, dimx*dimy*sizeof(float));
		// // for(int y=0; y<dimy; y++)
		// // {
			// // for(int x=0; x<dimx; x++)
			// // {
				// // if(
			// // }
		// // }
		// for(int k=0; k<total; k++)
		// {
			// if(ref[k] != arr[k])
			// {
				// cout << "Do not match at " << k << endl;
				// goto cleanup;
			// }
		// }
		// cout << "Matched!!!" << endl; 
		// cleanup:
		// free(ref);
		// free(arr);
	// }
	
	
 	MPI_Finalize();
	
	
	return 0;
}
	