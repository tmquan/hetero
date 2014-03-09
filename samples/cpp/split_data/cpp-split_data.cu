#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>
// #include <boost/program_options.hpp>
// #include <boost/algorithm/string.hpp>
#include "cmdparser.hpp"		// Input argument for C/C++ program
#include "datparser.hpp"		// Parsing the content of *dat file

#include <cuda.h>
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