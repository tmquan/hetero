#include <iostream> 
#include <hetero_cmdparser.hpp>
#include <cuda.h>
#include <helper_math.h>
#include <fstream>      
#include <sstream>      // std::istringstream
#include <limits>      // std::istringstream
#include <string.h>
#include <stdio.h>

using namespace std;
// -----------------------------------------------------------------------------------
#define checkReadFile(filename, pData, size) {                    				\
		fstream *fs = new fstream;												\
		fs->open(filename, ios::in|ios::binary);								\
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
// direction vectors
	const int dx[] = {-1,+0,+1,-1,+0,+1,-1,+0,+1, -1,+0,+1,-1,+1,-1,+0,+1, -1,+0,+1,-1,+0,+1,-1,+0,+1};
	const int dy[] = {-1,-1,-1,+0,+0,+0,+1,+1,+1, -1,-1,-1,+0,+0,+1,+1,+1, -1,-1,-1,+0,+0,+0,+1,+1,+1};
	const int dz[] = {-1,-1,-1,-1,-1,-1,-1,-1,-1, +0,+0,+0,+0,+0,+0,+0,+0, +1,+1,+1,+1,+1,+1,+1,+1,+1};
// -----------------------------------------------------------------------------------
void dfs(int x, int y, int z, int dimx, int dimy, int dimz, unsigned char *label, unsigned char current_label, unsigned char *mask) 
{
  if (x < 0 || x == dimx) return; // out of bounds
  if (y < 0 || y == dimy) return; // out of bounds
  if (z < 0 || z == dimz) return; // out of bounds
  if (label[at(x, y, z, dimx, dimy, dimz)] || (mask[at(x, y, z, dimx, dimy, dimz)] == 0)) return; // already labeled or not marked with 1 in m

  // mark the current cell
  // label[x][y] = current_label;
  label[at(x, y, z, dimx, dimy, dimz)] = current_label;

  // recursively mark the neighbors
  for (int direction = 0; direction < 26; ++direction)
    dfs(x+dx[direction], 
		y+dy[direction], 
		z+dz[direction], 
		dimx, dimy, dimz,
		label,
		current_label,
		mask);
}
// -----------------------------------------------------------------------------------
// void find_components() {
  // int component = 0;
  // for (int i = 0; i < row_count; ++i) 
    // for (int j = 0; j < col_count; ++j) 
      // if (!label[i][j] && m[i][j]) dfs(i, j, ++component);
// }
// -----------------------------------------------------------------------------------
int main(int argc, char **argv)
{
	const char* key =
		"{     |help      |       | print help message }"
		"{ i   |inputFile |       | inputFile}"
		"{ o   |outputFile|       | outputFile}"
		"{ x   |dimx      | 512   | dimx}"
		"{ y   |dimy      | 512   | dimy}"
		"{ z   |dimz      | 512   | dimz}";
		
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
	string outputFile		= cmd.get<string>("outputFile", true); //cout << outputDir << endl;
	const int    dimx		= cmd.get<int>("dimx", true);
	const int    dimy 		= cmd.get<int>("dimy", true);
	const int    dimz		= cmd.get<int>("dimz", true);
	const int 	 total 		= dimx*dimy*dimz;
	//----------------------------------------------------------------------------
	std::cout << "Minimum value for int: " << std::numeric_limits<int>::min() << '\n';
	std::cout << "Maximum value for int: " << std::numeric_limits<int>::max() << '\n';
	std::cout << "int is signed: " << std::numeric_limits<int>::is_signed << '\n';
	std::cout << "Non-sign bits in int: " << std::numeric_limits<int>::digits << '\n';
	std::cout << "int has infinity: " << std::numeric_limits<int>::has_infinity << '\n';
	//----------------------------------------------------------------------------
	// Create a buffer to contain the binary file
	unsigned char *mask;
	mask = new unsigned char[total];
	// mask = (unsigned char*)malloc(total*sizeof(unsigned char));
	// unsigned char mask[7773511680];
	checkReadFile(fileName.c_str(), mask, total*sizeof(unsigned char));
	    //----------------------------------------------------------------------------
    // fstream fs;
	// fs.open(fileName.c_str(), ios::in|ios::binary);								
	// if (!fs.good())														
	// {																		
		// cout << "Cannot open the file " << fileName << endl;				
        // fs.close();
		// return 1;															
	// }
	
	// fs.seekg(dimx*dimy*dimz*0, fs.beg); //set offset stridden from beginning
    // fs.read (reinterpret_cast<char*>(mask), total*sizeof(unsigned char));
	
	unsigned char *label;
	label = new unsigned char[total];
	// unsigned char label[7773511680];
	
	// // label = (unsigned char*)malloc(total*sizeof(unsigned char));
	
	unsigned char component = 128;
	for (int k=0; k<dimz; k++)
		for (int i=0; i<dimy; i++) 
			for (int j=0; j<dimx; j++) 
				if (!label[at(j,i,k,dimx,dimy,dimz)] && mask[at(j,i,k,dimx,dimy,dimz)]) 
				{
					dfs(j,i,k,dimx,dimy,dimz, label, ++component, mask);
					if(component==255)
						component = 128;
				}
			
	// Write to file
    checkWriteFile(outputFile.c_str(), label, total*sizeof(unsigned char));
	// free(mask);
	// free(label);
	return 0;
}