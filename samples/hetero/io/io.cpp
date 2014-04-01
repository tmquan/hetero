#include <iostream> 
#include <hetero_csv.hpp> //read_csv(argv[1], &fields, &values);
// #include <hetero_cmdparser.hpp>
#include <sstream>      // std::istringstream
#include <mpi.h>
using namespace std;

#define DIM_BUF 4
int main(int argc, char **argv)
{
    /// !!! Parsing the input file
    vector<string> fields;
    vector<string> values;
    
    read_csv(argv[1], &fields, &values);
    
    string objectFileName;
    int dimx;
    int dimy;
    int dimz;
    string type;
    
    for(int i=0; i<fields.size(); i++)
    {
        if(fields[i] == "objectFileName")   objectFileName  = values[i];
        if(fields[i] == "format")           type  = values[i];
        if(fields[i] == "dimx")             istringstream(values[i]) >> dimx;
        if(fields[i] == "dimy")             istringstream(values[i]) >> dimy;
        if(fields[i] == "dimz")             istringstream(values[i]) >> dimz;
    }
    
    cout << objectFileName << endl;
    cout << type << endl;
    cout << dimx << endl;
    cout << dimy << endl;
    cout << dimz << endl;
    
	///
	int rank, size;
	float buf[DIM_BUF];
	int offset, type_size;
	int i;
	MPI_File fh;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for(i=0;i<DIM_BUF;i++) buf[i] = rank*DIM_BUF+i;

	/* Open the file and write by using individual file pointers */
	// MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL,&fh);
	// MPI_Type_size(MPI_FLOAT, &type_size);
	// offset = rank*DIM_BUF*(type_size);
	// MPI_File_seek(fh,offset,MPI_SEEK_SET);
	// MPI_File_write(fh,buf,DIM_BUF,MPI_FLOAT,&status);
	// MPI_File_close(&fh);

	/* Re-open the file and read by using explicit offset */
	// MPI_File_open(MPI_COMM_WORLD,"output.dat",MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);
	MPI_Offset file_size;
	// MPI_File_get_size(fh,&file_size);
	// offset = file_size/size*rank;
	// printf("myid %d, filesize %lld, offset %d\n", rank,file_size,offset);
	// MPI_File_read_at(fh,offset,&buf,DIM_BUF,MPI_FLOAT,&status);

	// printf("myid %d, buffer after read:",rank);
	// for(i=0;i<DIM_BUF; i++)printf("%d ",buf[i]);
	// printf("\n\n");
	// MPI_File_close(&fh);

	/* Write the new file using the mpi_type_create_vector. Use the fileview */
	MPI_Datatype filetype;
	MPI_File_open(MPI_COMM_WORLD, "output_mod.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL,&fh);
	MPI_Type_vector(DIM_BUF/2,2,2*size,MPI_FLOAT,&filetype);
	MPI_Type_commit(&filetype);

	MPI_Type_size(MPI_FLOAT, &type_size);
	offset = 2*type_size*rank;
	MPI_File_set_view(fh,offset,MPI_FLOAT,filetype,"native",MPI_INFO_NULL);
	MPI_File_write_all(fh,buf,DIM_BUF,MPI_FLOAT,&status);
	MPI_File_get_size(fh,&file_size);
	printf("myid %d, filesize of the second file written %lld, offset %d\n", rank,file_size,offset);

	MPI_Type_free(&filetype);
	MPI_File_close(&fh);
	MPI_Finalize();
	return 0;
}