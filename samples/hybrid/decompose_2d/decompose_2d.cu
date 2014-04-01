#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string>

#include <mpi.h>
#include <cuda.h>

#include <assert.h>
#include <hetero_cmdparser.hpp>

using namespace std;
////////////////////////////////////////////////////////////////////////////////////////////////////

#define checkReadFile(filename, pData, size) {                    				\
		fstream *fs = new fstream;												\
		fs->open(filename.c_str(), ios::in|ios::binary);								\
		if (!fs->is_open())														\
		{																		\
			fprintf(stderr, "Cannot open file '%s' in file '%s' at line %i\n",	\
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
		fs->open(filename.c_str(), ios::out|ios::binary);								\
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
	// int i, j;
	
	// if(rank == master) 
	// {    
		// for(i=0; i<4; i++)
		// {
			// for(j=0; j<4; j++)  
			// {
				// coords[0] = i;
				// coords[1] = j;
				// MPI_Cart_rank(comm2d, coords, &rank);
				// printf("%d, %d, %d\n",coords[0],coords[1],rank);
			// }
		// }
	// }
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Retrieve the information from cmd
	const string srcFile  	= cmd.get<string>("srcFile", false);
	const int dimx  		= cmd.get<int>("dimx", false);
	const int dimy  		= cmd.get<int>("dimy", false);
	const int total 		= dimx*dimy;
	
	
	float *h_src 			= new float[total];
	
	// Read to pointer in master process
	if(rank==master)		checkReadFile(srcFile, h_src, total*sizeof(float)); 	
	
	int2 clusterDim = make_int2(dims[0], dims[1]);
	int  processIdx_1d = rank;
	int2 processIdx_2d = make_int2(coords[0], coords[1]);
	
	
	// #include <mpi.h>
	// int MPI_Scatterv(const void *sendbuf, const int sendcount[], const int displs[],
		// MPI_Datatype sendtype, void *recvbuf, int recvcount,
		// MPI_Datatype recvtype, int root, MPI_Comm comm)
	int *sendcount;    // array describing how many elements to send to each process
    int *displs;        // array describing the displacements where each segment begins
	
	sendcount 	= (int*)malloc(sizeof(int)*size);
    displs 		= (int*)malloc(sizeof(int)*size);
	
	 // calculate send counts and displacements
	int sum = 0;                // Sum of counts. Used to calculate displacements
    for (int i = 0; i < size; i++) {
        sendcount[i] = (dimx)/4;
        // if (rem > 0) {
            // sendcount[i]++;
            // rem--;
        // }
 
        displs[i] = sum;
        sum += sendcount[i];
    }
	
	// print calculated send counts and displacements for each process
    if(rank==master) 
	{
        for (int i = 0; i < size; i++) 
		{
            printf("sendcount[%d] = %d\tdispls[%d] = %d\n", i, sendcount[i], i, displs[i]);
        }
    }
	MPI_Barrier(MPI_COMM_WORLD);
	
	float *p_src = (float*)malloc(128*128*sizeof(float));
	MPI_Scatterv(h_src, sendcount, displs, MPI_FLOAT, 
		p_src, 128*128, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	
	/// Debug
	char *filename = new char[100];
	sprintf(filename, "result_%02d_%02d.raw", processIdx_2d.x, processIdx_2d.y);
	printf("%s\n", filename);
	
	stringstream ss;
	string s;
	
	ss << filename;
	ss >> s;
	checkWriteFile(s, p_src, 128*128*sizeof(float));
	
	
 	MPI_Finalize();
	return 0;
	
	// int MyRank, NewRank, Root=0, my_coords[2], dim_sizes[2], wrap_around[2], reorder;
	// MPI_Comm cart_comm;

	// MPI_Init(&argc,&argv);
	// MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

	// dim_sizes[0] = dim_sizes[1] = grid_side;
	// wrap_around[0] = wrap_around[1] = 0; //Non-periodic
	// reorder = 0; //False

	// MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, reorder, &cart_comm);
	// MPI_Comm_rank(cart_comm, &NewRank);

	// MPI_Cart_coords (cart_comm, NewRank, 2, my_coords);
	// printf ("Old Rank=%d, New Rank=%d, Coordinates=(%d,%d)\n", MyRank, NewRank, my_coords[0], my_coords[1]);

	// MPI_Finalize();
	// return 0;
}
	