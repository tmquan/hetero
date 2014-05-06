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
	MPI_Info info;

	MPI_Init(&argc, &argv);
	//================================================================================
	// Retrieve the number of execId
	// Parsing the arguments
	CommandLineParser cmd(argc, argv, key);
	const int dimx  			= cmd.get<int>("dimx", false);
	const int dimy  			= cmd.get<int>("dimy", false);
	const int dimz  			= cmd.get<int>("dimz", false);
	const string srcFile		= cmd.get<string>("srcFile", false);
	//================================================================================
	MPI_Info_create(&info);
	MPI_Info_set(info, "striping_factor", "4");
	MPI_Info_set(info, "striping_unit", "65536");
	//================================================================================
	
	MPI_Finalize();	
	return 0;
}