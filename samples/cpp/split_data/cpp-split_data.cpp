#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>
// #include <boost/program_options.hpp>
// #include <boost/algorithm/string.hpp>
#include "cmdparser.hpp"		// Input argument for C/C++ program
#include "datparser.hpp"		// Parsing the content of *dat file

#include <cuda.h>
#include <vector_types.h>		// For int3, dim3
// #include <helper_math.h>

using namespace std;

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
	
int main(int argc, char **argv)
{
	const char* key =
		"{     |help      |       | print help message }"
		"{ i   |input     |       | inputFile}";
		"{ h   |halo      |       | halo}";
		
	CommandLineParser cmd(argc, argv, key);
	if (argc == 1)
	{
		cout << "Usage: " << argv[0] << " [options]" << endl;
		cout << "Avaible options:" << endl;
		cmd.printParams();
		return 0;
	}
	
	cmd.printParams();
					  
	//----------------------------------------------------------------------------
	string fileName		= cmd.get<string>("input", true);
	int    halo			= cmd.get<int>("halo", true);
	//----------------------------------------------------------------------------
	DatFileParser dat(fileName);	
	dat.printContent();

	cout << endl;
	cout << dat.getObjectFileName() << endl;
	// cout << dat.getObjectFileName().size() << endl;
	cout << dat.getFileType() << endl;
	cout << dat.getdimx() << endl;
	cout << dat.getdimy() << endl;
	cout << dat.getdimz() << endl;
	//----------------------------------------------------------------------------
	const string objectFileName =  dat.getObjectFileName();
	cout << objectFileName << endl;
	const int dimx = dat.getdimx();
	const int dimy = dat.getdimy();
	const int dimz = dat.getdimz();
	
	const int chunkx   = 64;
	const int chunky   = 64;
	const int chunkz   = 64;
	//----------------------------------------------------------------------------
	// #pragma omp parallel for
	// for(int z=0; z<dimz; z++)
	// {
		// #pragma omp parallel for
		// for(int y=0; y<dimy; y++)
		// {
			// #pragma omp parallel for
			// for(int x=0; x<dimx; x++)
			// {
				// #pragma omp critical
				// {
					// // printf("Hello World from thread %02d out of %02d\n", thread, max_threads);
					// // cout << x << " " << y << " " << z << endl;
				// }
			// }
		// }
	// }
	
	
	unsigned char *pData;
	pData = new unsigned char[64];
	//----------------------------------------------------------------------------
	fstream fs;
	fs.open(objectFileName, ios::in|ios::binary);								
	if (fs.good())														
	{																		
		// fprintf("Cannot open file '%s' in file '%s' at line %i\n",	
			// objectFileName, __FILE__, __LINE__);										
		printf("Cannot open file %s\n", objectFileName);										
		return 1;															
	}
	//----------------------------------------------------------------------------
	// int3 numThreads(64, 64, 64);
	// int3 numBlocks(64, 64, 64);
	
	dim3 numThreads(64, 64, 64);
	dim3 numBlocks((dimx/64 + ((dimx%64)?1:0)),
				   (dimy/64 + ((dimy%64)?1:0)),
				   (dimz/64 + ((dimz%64)?1:0)) );
	dim3 blockDim(64, 64, 64);
	dim3 gridDim((dimx/64 + ((dimx%64)?1:0)),
				   (dimy/64 + ((dimy%64)?1:0)),
				   (dimz/64 + ((dimz%64)?1:0)) );

	// int3 blockIdx = make_int3(0, 0, 0);
	// int3 threadIdx = make_int3(0, 0, 0);
	int3 blockIdx {0, 0, 0};
	int3 threadIdx{0, 0, 0};
	
	cout << blockDim.x << " " << blockDim.y << " " << blockDim.z << endl;
	cout << blockIdx.x << " " << blockIdx.y << " " << blockIdx.z << endl;
	
	int3 global_index_3d{0, 0, 0};
	int  global_index_1d = 0;
	
	// #pragma omp parallel for
	for(blockIdx.z=0; blockIdx.z<gridDim.z; blockIdx.z++)
	{
		for(blockIdx.y=0; blockIdx.y<gridDim.y; blockIdx.y++)
		{
			for(blockIdx.x=0; blockIdx.x<gridDim.x; blockIdx.x++)
			{
				//cout << blockIdx.x << " " << blockIdx.y << " " << blockIdx.z << endl;
				for(threadIdx.z=0; threadIdx.z<blockDim.z; threadIdx.z++)
				{
					for(threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y++)
					{
						for(threadIdx.x=0; threadIdx.x<blockDim.x; threadIdx.x++)
						{
							global_index_3d.x = blockDim.x * blockIdx.x + threadIdx.x;
							global_index_3d.y = blockDim.y * blockIdx.y + threadIdx.y;
							global_index_3d.z = blockDim.z * blockIdx.z + threadIdx.z;
							global_index_1d   = global_index_3d.z * dimy * dimx +
											    global_index_3d.y * dimx +
											    global_index_3d.x;
							// cout << global_index_1d << endl;
							if(global_index_3d.x < dimx && 
							   global_index_3d.y < dimy &&
							   global_index_3d.z < dimz)
							{
								
							}
						}
						goto debug;
					}
				}
		
			}
			// goto debug;
		}
	}
debug:
	// for(int cx=0; cx<chunkx; cx++)
	// {
		// printf("%u\t", pData[cx]);
	// }
	// printf("\n");

	
	//----------------------------------------------------------------------------
	// ! Read here
	// fs.read(reinterpret_cast<char*>(pData), chunkx*sizeof(unsigned char));
	// for(int cx=0; cx<chunkx; cx++)
		// printf("%u\t", pData[cx]);
	// printf("\n");
	for(int cy=0; cy<chunky; cy++)
	{
		// printf("%u\t", pData[cx]);
		// fs.seekg(
		fs.read(reinterpret_cast<char*>(pData), chunkx*sizeof(unsigned char));
	}	
	printf("\n");
	
	//----------------------------------------------------------------------------
	// streampos begin,end;
	// begin = fs.tellg();
	// fs.seekg(0, ios::end);
	// end = fs.tellg();
	// fs.close();															
	// cout << "size is: " << (end-begin) << " bytes.\n";

	//----------------------------------------------------------------------------
	// fs->close();															
	
	// // delete fs;	

	return 0;
}