#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <omp.h>
// #include <boost/program_options.hpp>
// #include <boost/algorithm/string.hpp>
#include "cmdparser.hpp"		// Input argument for C/C++ program
#include "datparser.hpp"		// Parsing the content of *dat file
#include "cpu_timer.hpp"		// Parsing the content of *dat file

#include <cuda.h>
#include <vector_types.h>		// For int3, dim3
#include "helper_math.h"

using namespace std;
// -----------------------------------------------------------------------------------
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
// -----------------------------------------------------------------------------------
/// Mirror effect, acts like Neumann Boundary Condition
#define at(x, y, z, dimx, dimy, dimz) (clamp(z, 0, dimz-1)*dimy*dimx		\
									+clamp(y, 0, dimy-1)*dimx				\
									+clamp(x, 0, dimx-1))		
// -----------------------------------------------------------------------------------
int main(int argc, char **argv)
{
	const char* key =
		"{     |help      |       | print help message }"
		"{ i   |inputFile |       | inputFile}"
		"{ o   |outputDir | ./    | outputDir}"
		"{ x   |numx      | 2     | divide along x}"
		"{ y   |numy      | 2     | divide along y}"
		"{ z   |numz      | 2     | divide along z}"
		"{ h   |halo      | 1     | halo}";
		
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
	string fileName			= cmd.get<string>("inputFile", true);
	string outputDir		= cmd.get<string>("outputDir", true); //cout << outputDir << endl;
	int    numx				= cmd.get<int>("numx", true);
	int    numy 			= cmd.get<int>("numy", true);
	int    numz				= cmd.get<int>("numz", true);
	int    halo				= cmd.get<int>("halo", true);
	//----------------------------------------------------------------------------
	DatFileParser dat(fileName);	
	dat.printContent();

	// cout << endl;
	// cout << dat.getObjectFileName() << endl;
	// // cout << dat.getObjectFileName().size() << endl;
	// cout << dat.getFileType() << endl;
	// cout << dat.getdimx() << endl;
	// cout << dat.getdimy() << endl;
	// cout << dat.getdimz() << endl;
	//----------------------------------------------------------------------------
	const string objectFileName =  dat.getObjectFileName();
	// cout << objectFileName << endl;
	const int dimx = dat.getdimx();
	const int dimy = dat.getdimy();
	const int dimz = dat.getdimz();
    // Print info of dat file during the program, not the file itself
    cout << "Print info of dat file during the program, not the file itself" << endl;
    cout << objectFileName << endl;
    cout << dimx << endl;
    cout << dimy << endl;
    cout << dimz << endl;
	
    
    //----------------------------------------------------------------------------
    fstream fs;
	fs.open(objectFileName, ios::in|ios::binary);								
	if (!fs.good())														
	{																		
		cout << "Cannot open the file " << objectFileName << endl;				
        fs.close();
		return 1;															
	}
	
	// fs.open(objectFileName, ios::in|ios::binary);								
	//----------------------------------------------------------------------------
	// Determine the size of 1 element
	// if(fs)
	// {
		fs.seekg(0, fstream::end);
		long length = fs.tellg();
		cout << "Length: " << length << endl;
		fs.seekg(0, fstream::beg); // Reset at the begining
	// }
	//----------------------------------------------------------------------------
	
	// int3 numThreads(64, 64, 64);
	// int3 numBlocks(64, 64, 64);
	
	// dim3 numThreads(64, 64, 64);
	// dim3 numBlocks((dimx/64 + ((dimx%64)?1:0)),
				   // (dimy/64 + ((dimy%64)?1:0)),
				   // (dimz/64 + ((dimz%64)?1:0)) );
	// dim3 blockDim(64, 64, 64);
	// dim3 gridDim((dimx/64	 + ((dimx%64)?1:0)),
                 // (dimy/64 + ((dimy%64)?1:0)),
                 // (dimz/64 + ((dimz%64)?1:0)) );
    dim3 gridDim(numx, numy, numz);
    dim3 blockDim((dimx/numx + ((dimx%numx)?1:0)),
                  (dimy/numy + ((dimy%numy)?1:0)),
                  (dimz/numz + ((dimx%numz)?1:0)));
                  
    dim3 blockDimWithHalo(blockDim.x + 2*halo,
                          blockDim.y + 2*halo,
                          blockDim.z + 2*halo);
                         
	// int3 blockIdx = make_int3(0, 0, 0);
	// int3 threadIdx = make_int3(0, 0, 0);
	int3 blockIdx {0, 0, 0};
	int3 threadIdx{0, 0, 0};
	
	cout << blockDim.x << " " << blockDim.y << " " << blockDim.z << endl;
	cout << blockIdx.x << " " << blockIdx.y << " " << blockIdx.z << endl;
	
	int3 shared_index_3d{0, 0, 0};
	int  shared_index_1d = 0;
	
	int3 global_index_3d{0, 0, 0};
	int  global_index_1d = 0;

	int3 index_3d{0, 0, 0};
	int  index_1d = 0;
	
	int global_block_index_1d;
	// string outputFileName;
	char outputFileName[100];
	unsigned char *blockData;
	// blockData = new unsigned char[blockDim.x*blockDim.y*blockDim.z];
    // -----------------------------------------------------------------------------------    
    //!!! Quan's algorithm: Multi batch loading with halo
    // Continuous memory transaction
    // First calculate the number of batch, round of the result from (blockSize with halo)/(blockSize without halo)
    int blockSizeWithHalo    = (blockDim.x + 2*halo) * (blockDim.y + 2*halo) * (blockDim.z + 2*halo); 
    int blockSize            = (blockDim.x + 0*halo) * (blockDim.y + 0*halo) * (blockDim.z + 0*halo);
    
    
    
    int numBatches = (blockSizeWithHalo/blockSize + ((blockSizeWithHalo%blockSize)?1:0));
    int batch = 0; //For reading below
    
    blockData = new unsigned char[blockSizeWithHalo];
    
    cout << "BlockSizeWithHalo: " << blockSizeWithHalo << endl;
    cout << "BlockSize        : " << blockSize         << endl;
    cout << "Number of batches: " << numBatches        << endl;
    
    // #pragma omp parallel for
	CpuTimer cpu_timer;
	cpu_timer.Start();
	// #pragma omp parallel 
	for(blockIdx.z=0; blockIdx.z<gridDim.z; blockIdx.z++)
	{
		// #pragma omp parallel 
		for(blockIdx.y=0; blockIdx.y<gridDim.y; blockIdx.y++)
		{
			// #pragma omp parallel 
			for(blockIdx.x=0; blockIdx.x<gridDim.x; blockIdx.x++)
			{
                memset(blockData, 0, blockSizeWithHalo*sizeof(unsigned char)); // need #include <string.h>
				// #pragma omp parallel 
				for(threadIdx.z=0; threadIdx.z<blockDim.z; threadIdx.z++)
				{
					// #pragma omp parallel 
					for(threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y++)
					{
						// #pragma omp parallel 
						
						for(threadIdx.x=0; threadIdx.x<blockDim.x; threadIdx.x++)
						{
                            
                            // -----------------------------------------------------------------------------------    
                            for(batch=0; batch<numBatches; batch++)
                            {
                                shared_index_1d 	= threadIdx.z * blockDim.y * blockDim.x +
                                                      threadIdx.y * blockDim.x + 
                                                      threadIdx.x +
                                                      blockDim.x  * blockDim.y * blockDim.z * batch;  // Next number of loading, magic is here
                                                      
                                shared_index_3d.x   = (shared_index_1d % (blockDimWithHalo.y * blockDimWithHalo.x)) % blockDimWithHalo.x;
                                shared_index_3d.y   = (shared_index_1d % (blockDimWithHalo.y * blockDimWithHalo.x)) / blockDimWithHalo.x;
                                shared_index_3d.z   = (shared_index_1d / (blockDimWithHalo.y * blockDimWithHalo.x));
                                
                                global_index_3d.x   = blockIdx.x * blockDim.x + shared_index_3d.x - halo;
                                global_index_3d.y   = blockIdx.y * blockDim.y + shared_index_3d.y - halo;
                                global_index_3d.z   = blockIdx.z * blockDim.z + shared_index_3d.z - halo;
                                
                                global_index_1d 	= global_index_3d.z * dimy * dimx + 
                                                      global_index_3d.y * dimx + 
                                                      global_index_3d.x;
                                if (shared_index_3d.z < blockDimWithHalo.z) 
                                {
                                    // global_index_3d.z = clamp(global_index_3d.z, 0, dimz-1);
                                    // global_index_3d.y = clamp(global_index_3d.y, 0, dimy-1);
                                    // index_3d.x   = blockIdx.x * blockDim.x + threadIdx.x;
                                    // index_3d.y   = blockIdx.y * blockDim.y + threadIdx.y;
                                    // index_3d.z   = blockIdx.z * blockDim.z + threadIdx.z;
                                
                                    // index_1d 	 = index_3d.z * dimy * dimx + 
                                                   // index_3d.y * dimx + 
                                                   // index_3d.x;
                                    // if (global_index_3d.z >= 0 && global_index_3d.z < dimz && 
                                        // global_index_3d.y >= 0 && global_index_3d.y < dimy &&
                                        // global_index_3d.x >= 0 && global_index_3d.x < dimx )	
                                        // // sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = src[global_index_1d];
                                    // {
                                        // // // Read one element
                                        // // fs.seekg(global_index_1d, fs.beg); //set offset stridden from beginning
                                        // // fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), sizeof(unsigned char)); 
                                        // // Read blockDim.x elements, a stride of memory: 140 sec to 11 sec
                                        // if(shared_index_3d.x==0)
                                        // {                                                     
                                            // fs.seekg(global_index_1d, fs.beg);
                                            // // if(blockIdx.x<(blockDim.x-1))
                                            // // fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), (blockDim.x+2*halo) * sizeof(unsigned char));
                                            // // else
                                            // fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), (blockDim.x+1*halo) * sizeof(unsigned char));
                                        // }
                                    // }
                                    // else if (global_index_3d.x < 0
                                    // // || (global_index_3d.x >dimz && global_index_3d.x <=(dimz+halo))
                                    // )	
                                        // // sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = src[global_index_1d];
                                    // {
                                        // // // Read one element
                                        // // fs.seekg(global_index_1d, fs.beg); //set offset stridden from beginning
                                        // // fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), sizeof(unsigned char)); 
                                        // // Read blockDim.x elements, a stride of memory: 140 sec to 11 sec
                                        // if(shared_index_3d.x==0)
                                        // {             
                                            // // global_index_3d.x = clamp(global_index_3d.x, 0, dimy-1);
                                            // // global_index_3d.z = clamp(global_index_3d.z, 0, dimz-1);
                                            // // global_index_3d.y = clamp(global_index_3d.y, 0, dimy-1);
                                            
                                            // // global_index_1d   = global_index_3d.z * dimy * dimx + 
                                                                // // global_index_3d.y * dimx + 
                                                                // // global_index_3d.x;
                                            // global_index_1d = clamp(global_index_1d, 0, length);         
                                            // fs.seekg(global_index_1d, fs.beg);
                                            // fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), (blockDim.x+2*halo) * sizeof(unsigned char));
                                        // }
                                    // }
                                    if (global_index_3d.z >= 0 && global_index_3d.z < dimz && 
                                        global_index_3d.y >= 0 && global_index_3d.y < dimy &&
                                        global_index_3d.x >= 0 && global_index_3d.x < dimx )	
                                        // sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = src[global_index_1d];
                                    {
                                        // // Read one element
                                        fs.seekg(global_index_1d, fs.beg); //set offset stridden from beginning
                                        fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), sizeof(unsigned char)); 
                                        // // Read blockDim.x elements, a stride of memory: 140 sec to 11 sec
                                        // if(shared_index_3d.x==0)
                                        // {                                                     
                                            // fs.seekg(global_index_1d, fs.beg);
                                            // fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), (blockDim.x+1*halo) * sizeof(unsigned char));
                                        // }
                                    }
                                    else
                                        // sharedMem[shared_index_3d.z][shared_index_3d.y][shared_index_3d.x] = -1;
                                    {
                                        blockData[shared_index_1d] = 0; //Actually, it is initialized with 0 (memset), can be skipped
                                    }
                                }
                                // __syncthreads();
                            }
                            // -----------------------------------------------------------------------------------    
							// shared_index_3d.x = threadIdx.x;
							// shared_index_3d.y = threadIdx.y;
							// shared_index_3d.z = threadIdx.z;
                            
							// shared_index_1d   = shared_index_3d.z * blockDim.y * blockDim.x +
												// shared_index_3d.y * blockDim.x +
												// shared_index_3d.x;
												
							// global_index_3d.x = blockDim.x * blockIdx.x + threadIdx.x;
							// global_index_3d.y = blockDim.y * blockIdx.y + threadIdx.y;
							// global_index_3d.z = blockDim.z * blockIdx.z + threadIdx.z; 
                            
							// global_index_1d   = global_index_3d.z * dimy * dimx +
											    // global_index_3d.y * dimx +
											    // global_index_3d.x;
							// // cout << global_index_1d << endl;
							// if(global_index_3d.x < dimx && 
							   // global_index_3d.y < dimy &&
							   // global_index_3d.z < dimz)
							// {
								// // Set offset in here, basically base on the tellg and seekg
								// //---------------------------------------------
								// // Read one element
								// // fs.seekg(global_index_1d, fs.beg);
								// // fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), sizeof(unsigned char));
								
								// // Read blockDim.x elements, a stride of memory: 128 sec to 4 sec
								// // if(threadIdx.x==0)
								// // {
									// // fs.seekg(global_index_1d, fs.beg);
									// // fs.read (reinterpret_cast<char*>(blockData+shared_index_1d), blockDim.x * sizeof(unsigned char));
								// // }
							// }
						}						
						// goto debug;
					}
				}
				// checkWriteFile(outputDir, blockData, blockDim.x*blockDim.y*blockDim.z*sizeof(unsigned char));
				cout << "Size of uchar: " << sizeof(unsigned char) << endl;
				cout << "Size of float: " << sizeof(float) << endl;
				cout << outputDir << endl;
				global_block_index_1d = blockIdx.z * gridDim.y * gridDim.x +
									    blockIdx.y * gridDim.x +
										blockIdx.x;
				sprintf(outputFileName, "%04d.raw", global_block_index_1d);
				cout << outputFileName << endl;
				// checkWriteFile(outputDir+"test.raw", blockData, blockDim.x*blockDim.y*blockDim.z*sizeof(unsigned char));
				// checkWriteFile(outputDir+outputFileName, blockData, blockDim.x*blockDim.y*blockDim.z*sizeof(unsigned char));
				checkWriteFile(outputDir+outputFileName, blockData, blockSizeWithHalo*sizeof(unsigned char));
				// delete blockData;
				// goto debug;
			}
			
		}
	}
	cpu_timer.Stop();
	cout << "Elapsed Time (ms): " << cpu_timer.Elapsed() << endl;
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
	// for(int cy=0; cy<chunky; cy++)
	// {
		// printf("%u\t", pData[cx]);
		// fs.seekg(
		// fs.read(reinterpret_cast<char*>(pData), chunkx*sizeof(unsigned char));
	// }	
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