#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	int size, rank;
	char name[MPI_MAX_PROCESSOR_NAME];
	int length;
		
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &length);
	
  	printf("This is rank %02d, size %02d, of %s\n", rank, size, name);

  	MPI_Finalize();
	return 0;
}