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
	"{ halo|halo      |      | halo size   }"
	"{ dimx|dimx      |      | dimension x }"
	"{ dimy|dimy      |      | dimension y }"
	"{ dimz|dimz      |      | dimension z }"
	;
////////////////////////////////////////////////////////////////////////////////////////////////////
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

	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	
  	printf("This is rank %02d, size %02d, of %s\n", rank, size, name);
	MPI_Barrier(MPI_COMM_WORLD);
	//-----------------------------------------------------------------------------------------
	MPI_Comm comm3d;          		/* Cartesian communicator */
    int dims[3];// = {0, 0};   		/* allow MPI to choose grid block dimensions */
    int periodic[3];// = {0, 0};  	/* domain is non-periodic */
    int reorder;// = 1;          	/* allow processes to be re-ranked */
    int coords[3];            		/* coordinates of our block in grid */
    // int up, down;             		/* ranks of processes above and below ours */
    // int left, right;          		/* ranks of processes to each side of ours */
	
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
	dims[2] = 2;
	periodic[0] = 0;
	periodic[1] = 0;
	periodic[2] = 0;
	reorder = 1;
	// // Set up Cartesian grid of processors.  A new communicator is
    // // created we get our rank within it. 
    // MPI_Dims_create(rank, 2, dims); ///This line will not work
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, reorder, &comm3d );
    MPI_Cart_get(comm3d, 3, dims, periodic, coords );
    MPI_Comm_rank(comm3d, &rank );
	printf("x %d, y %d, z %d, rank %d\n",coords[0],coords[1],coords[2],rank);

	MPI_Barrier(MPI_COMM_WORLD);
	//-----------------------------------------------------------------------------------------
	// Retrieve the information from cmd
	string srcFile  	= cmd.get<string>("srcFile", false);
	const int halo  	= cmd.get<int>("halo", false);
	const int dimx  	= cmd.get<int>("dimx", false);
	const int dimy  	= cmd.get<int>("dimy", false);
	const int dimz  	= cmd.get<int>("dimz", false);
	const int total 	= dimx*dimy*dimz;
	
	
	// float *h_src 			= new float[total];
	int  processIdx_1d 	= rank;
	int3 processIdx_3d 	= make_int3(coords[0], coords[1], coords[2]);
	
	
	

	/// Mimic Pack and Unpack MPI

	int3 featureIdx		{  0,   0,	0};
	int3 processIdx		{  0,   0,	0};
	int3 processDim		{  256,   256,  256};
	int3 subDataDim		{0, 0, 0};
	int3 clusterDim    	{(dimx/processDim.x + ((dimx%processDim.x)?1:0)),
						 (dimy/processDim.y + ((dimy%processDim.y)?1:0)),
						 (dimz/processDim.z + ((dimz%processDim.z)?1:0))};
	MPI_Barrier(MPI_COMM_WORLD);
	// cout << "Cluster Dimension: " << clusterDim.x << " "  
								  // << clusterDim.y << " " 
								  // << clusterDim.z << " "
								  // << endl;		
	MPI_Barrier(MPI_COMM_WORLD);
	//-----------------------------------------------------------------------------------------
	MPI_Request request;
	MPI_Request status;
	//Start packing
	/// Naive approach, copy to another buffer, then send
	int3 index_3d;
	double start = MPI_Wtime();
	int caught = 0;
	if(rank==master)
	{
		for(processIdx.z=0; processIdx.z<clusterDim.z; processIdx.z++)
		{
			for(processIdx.y=0; processIdx.y<clusterDim.y; processIdx.y++)
			{
				for(processIdx.x=0; processIdx.x<clusterDim.x; processIdx.x++)
				{
					/// !!! First step: Determine size of buffer
					for(featureIdx.z=0; featureIdx.z<processDim.z; featureIdx.z++)
					{
						for(featureIdx.y=0; featureIdx.y<processDim.y; featureIdx.y++)
						{
							for(featureIdx.x=0; featureIdx.x<processDim.x; featureIdx.x++)
							{
								//3D global index
								index_3d = make_int3(
									processIdx.x*processDim.x+featureIdx.x,
									processIdx.y*processDim.y+featureIdx.y,
									processIdx.z*processDim.z+featureIdx.z);	
								if(index_3d.x==dimx) break;
							}
							if(index_3d.y==dimy) break;
						}	
						if(index_3d.z==dimz) break;
					}
					subDataDim = make_int3(featureIdx.x, featureIdx.y, featureIdx.z);
					cout << "Sub problem size: " << subDataDim.x << " "  << subDataDim.y << " "  << subDataDim.z << endl;
					
					//Second step: copy subdataSize
					index_3d = make_int3(
						processIdx.x*processDim.x+0,
						processIdx.y*processDim.y+0, 	
						processIdx.z*processDim.z+0);	
					MPI_Datatype mysubarray;
					int starts[3] 	 = {index_3d.z, index_3d.y, index_3d.x}; ///!Order is very important
					int subsizes[3]  = {subDataDim.z, subDataDim.y, subDataDim.x}; ///!Order is very important
					int bigsizes[3]  = {dimz, dimy, dimx}; ///!Order is very important
					MPI_Type_create_subarray(3, bigsizes, subsizes, starts,
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

					processIdx_1d = processIdx.z * clusterDim.y * clusterDim.x + 
									processIdx.y * clusterDim.x + 
									processIdx.x;
					cout << processIdx_1d << endl;
					
					/// !!! Send to worker process
					// Send the size of message
					MPI_Isend(&subDataDim, 1, MPI_LONG_DOUBLE, processIdx_1d, 0, MPI_COMM_WORLD, &request);	//Data need to be long enough
					
					cout << "Sent" << endl;
					
					
					MPI_Type_free(&mysubarray);
				}
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Recv(p_src, processDim.x*processDim.y, MPI_FLOAT, master, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&subDataDim, 1, MPI_LONG_DOUBLE, master, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	// MPI_Irecv(&subDataDim, 1, MPI_LONG_DOUBLE, master, 0, MPI_COMM_WORLD, &request);
	// MPI_Wait(&request, &status);
	//MPI_Recv(p_src, subDataDim.x *  subDataDim.y, MPI_FLOAT, master, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	cout << "Receive preamble read from " << rank << endl;
	cout << "Sub problem size: " << subDataDim.x << " "  << subDataDim.y << " "  << subDataDim.z << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	
	float *p_src;
	p_src = (float*)malloc(subDataDim.x*subDataDim.y*subDataDim.z*sizeof(float));
	//-----------------------------------------------------------------------------------------
	int npx = dimx/processDim.x + 2*halo;
	int npy = dimy/processDim.y + 2*halo;
	int npz = dimz/processDim.z + 2*halo;
	
	// if(rank==0)
	// {
		// cout << "At " << rank << endl;
		// cout << npx << " " <<  npy << " " << npz << endl;
	// }
	//-----------------------------------------------------------------------------------------
	// // Construct the neighbor communicator
	// int left, right, top, bottom, front, back;
	// MPI_Cart_shift(comm3d, 1, 1, &left, &right);
	// MPI_Cart_shift(comm3d, 2, 1, &top, &bottom);
	// MPI_Cart_shift(comm3d, 0, 1, &front, &back);
	// MPI_Barrier(MPI_COMM_WORLD);
	// fprintf(stderr, "Rank %d has LR neighbours %d %d, FB %d %d, TB %d %d\n", 
			// rank, left, right, front, back, top, bottom);
	// MPI_Barrier(MPI_COMM_WORLD);
	//-----------------------------------------------------------------------------------------
	// // create subarrays(exclude halo) to write to file with MPI-IO 
	// // data in the local array 
	// int sizes[3];
	// sizes[0]=npz; sizes[1]=npx; sizes[2]=npy;
	
	// int subsizes[3];
	// subsizes[0]=sizes[0]-2*halo;	subsizes[1]=sizes[1]-2*halo;	subsizes[2]=sizes[2]-2*halo;
	
	// int starts[3];
	// starts[0]=halo; starts[1]=halo; starts[2]=halo;
	
	// MPI_Datatype local_array;
	// MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &local_array);
	// MPI_Type_commit(&local_array);
	//-----------------------------------------------------------------------------------------
	// // data in the global array 
	// int gsizes[3];
	// // gsizes[0]=nz; gsizes[1]=nx; gsizes[2]=ny;
	// gsizes[0]=dimz; gsizes[1]=dimx; gsizes[2]=dimy;
	// int gstarts[3];
	// gstarts[0]=subsizes[0]*coords[0]; 	gstarts[1]=subsizes[1]*coords[1];	gstarts[2]=subsizes[2]*coords[2];
	
	// MPI_Datatype global_array;
	// MPI_Type_create_subarray(3, gsizes, subsizes, gstarts, MPI_ORDER_C,	MPI_FLOAT, &global_array);
	// MPI_Type_commit(&global_array);
	//-----------------------------------------------------------------------------------------
	// /* allocate of halo areas */
	// int   halosizex = npy*npz*halo;
	// float *leftRecv  = (float *)calloc(3*halosizex,sizeof(float));
	// float *rightRecv = (float *)calloc(3*halosizex,sizeof(float));
	// float *leftSend  = (float *)calloc(3*halosizex,sizeof(float));
	// float *rightSend = (float *)calloc(3*halosizex,sizeof(float));

	// int   halosizey = npx*npz*halo;
	// float *frontRecv = (float *)calloc(3*halosizey,sizeof(float));
	// float *backRecv  = (float *)calloc(3*halosizey,sizeof(float));
	// float *frontSend = (float *)calloc(3*halosizey,sizeof(float));
	// float *backSend  = (float *)calloc(3*halosizey,sizeof(float));

	// int   halosizez = npy*npx*halo;
	// float *topRecv    = (float *)calloc(3*halosizez,sizeof(float));
	// float *bottomRecv = (float *)calloc(3*halosizez,sizeof(float));
	// float *topSend    = (float *)calloc(3*halosizez,sizeof(float));
	// float *bottomSend = (float *)calloc(3*halosizez,sizeof(float));
	//-----------------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------
				char *ch = strdup(srcFile.c_str());

				cout << ch << endl;
				
				MPI_Offset disp;
				disp = sizeof(float)*rank*processDim.x*processDim.y*processDim.z; 
				MPI_Datatype etype;
				etype = MPI_FLOAT;
				
				index_3d = make_int3(
					(rank%(2*2)%2)*processDim.x+0,
					(rank%(2*2)/2)*processDim.y+0,
					(rank/(2*2))*processDim.z+0);	
				// index_3d = make_int3(
					// (coords[1])*processDim.x+0,
					// (coords[2])*processDim.y+0,
					// (coords[0])*processDim.z+0);	
				// index_2d = make_int2(
					// (rank%2)*subDataDim.x+0,
					// (rank/2)*subDataDim.y+0);
					
				cout << "Start read from " << rank << endl;
				int bigsizes[3]  = {dimz, dimy, dimx}; 									///!Order is very important
				int subsizes[3]  = {subDataDim.z, subDataDim.y, subDataDim.x}; 			///!Order is very important
				int starts[3] 	 = {index_3d.z, index_3d.y, index_3d.x}; 					///!Order is very important
					
				MPI_Barrier(MPI_COMM_WORLD);
				cout << "Start indices \t" << index_3d.x << " \t" << index_3d.y << " \t" << index_3d.z << " \t at " << rank << endl;
				MPI_Barrier(MPI_COMM_WORLD);
				
				MPI_Datatype subarray;
				MPI_Type_create_subarray(3, bigsizes, subsizes, starts,
					MPI_ORDER_C, MPI_FLOAT, &subarray);
				
				MPI_Type_commit(&subarray);
				errCode = MPI_File_open(MPI_COMM_WORLD, ch,	MPI_MODE_RDONLY,  MPI_INFO_NULL, &fh);
				cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;		
				if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
				
				// MPI_File_set_view(fh, disp, etype, subarray, "native", MPI_INFO_NULL);
				MPI_File_set_view(fh, 0, etype, subarray, "native", MPI_INFO_NULL);
				cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
				
				MPI_File_read(fh, p_src, subDataDim.x*subDataDim.y*subDataDim.z, MPI_FLOAT, MPI_STATUS_IGNORE); 
				// MPI_File_read_all(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); // Process spawn and fail
				// MPI_File_read_ordered(fh, p_src, subDataDim.x*subDataDim.y, MPI_FLOAT, MPI_STATUS_IGNORE); 
				// MPI_Type_free(&subarray);
				cout << "Debug at " << __FILE__ << " " << __LINE__ << endl;	
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
				sprintf(filename, "result_%02d_%02d_%02d.raw", processIdx_3d.x, processIdx_3d.y, processIdx_3d.z);
				printf("%s\n", filename);
				// float *h_tmp;
				// h_tmp = (float*)malloc(subDataDim.x*subDataDim.y*sizeof(float));
				// cudaHostRegister(h_tmp, subDataDim.x*subDataDim.y *sizeof(float), cudaHostRegisterPortable);
				// cudaMemcpy(h_tmp, p_src, subDataDim.x*subDataDim.y*sizeof(float), cudaMemcpyDeviceToHost); cudaCheckLastError();
				// checkWriteFile(filename, h_tmp, subDataDim.x*subDataDim.y*sizeof(float));
				// checkWriteFile(filename, p_src, processDim.x*processDim.y*sizeof(float));
				checkWriteFile(filename, p_src, subDataDim.x*subDataDim.y*subDataDim.z*sizeof(float));
				
				
				///!!! Write globally
				
				// Delete the file before using that
				MPI_Barrier(MPI_COMM_WORLD);	
				if(rank == master)
				{
					// errCode = MPI_File_delete("test.raw", MPI_INFO_NULL);
					// if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_delete");
				}
				MPI_Barrier(MPI_COMM_WORLD);	
				
				errCode = MPI_File_open(MPI_COMM_WORLD, "test.raw",	MPI_MODE_RDWR|MPI_MODE_CREATE,  MPI_INFO_NULL, &fh);
				
				if (errCode != MPI_SUCCESS) handle_error(errCode, "MPI_File_open");
				
				MPI_File_set_view(fh, 0, etype, subarray, "native", MPI_INFO_NULL);
				MPI_Type_free(&subarray);
				
				cout << "At rank " << rank << endl;
				cout << "Sub problem size will be written: " << subDataDim.x << " "  
															 << subDataDim.y << " " 
															 << subDataDim.z << endl;
				MPI_File_write_all(fh, p_src, subDataDim.x*subDataDim.y*subDataDim.z, MPI_FLOAT, MPI_STATUS_IGNORE); 
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
	//-----------------------------------------------------------------------------------------
	
	
	MPI_Finalize();	
	return 0;
}