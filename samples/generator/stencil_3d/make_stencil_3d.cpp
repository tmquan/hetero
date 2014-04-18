#include <iostream>
#include <fstream>
#include <map>
#include <string>

using namespace std;
class heteroGenerator3D
{
public:
	heteroGenerator3D(void);
	heteroGenerator3D(string);
	~heteroGenerator3D(void);	
	
	void addSrcArray(pair<string, string>);
	void addDstArray(pair<string, string>);
	void addAttribute(pair<string, string>);
	void addSharedMem(pair<string, string>);
	void addVariable(pair<string, string>);
	void addRegister(pair<string, string>);
	
	void generateHeaderFile();
	void generateKernelFile();

private:
	string module;
	string strHppFile ;
	string strCppFile ;
	string strCUDAFile;
	
	fstream fs;		
	
	string strKernel; //Contain the stencil code only	
	string dataType;
	map <string, string> 	_srcArrayList; //name and type
	map <string, string> 	_dstArrayList;
	map <string, string> 	_attributeList;
	map <string, string> 	_sharedMemList;
	map <string, string> 	_variableList;
	map <string, string> 	_registerList;
};	
                  
heteroGenerator3D::heteroGenerator3D()
{
	module 			= "function";
	strHppFile 		= module+".hpp";
	strCppFile 		= module+".cpp";
	strCUDAFile		= module+".cu";
	
	_srcArrayList.clear();
	_dstArrayList.clear();
    _attributeList.clear();
    _sharedMemList.clear();	
    _registerList.clear();	
	
	this->addAttribute(make_pair("dimx","int"));
	this->addAttribute(make_pair("dimy","int"));
}

heteroGenerator3D::heteroGenerator3D(string __module)
{
	module 			= __module;
	strHppFile 		= module+".hpp";
	strCppFile 		= module+".cpp";
	strCUDAFile		= module+".cu";
	
	_srcArrayList.clear();
	_dstArrayList.clear();
    _attributeList.clear();
    _sharedMemList.clear();
	_registerList.clear();	
	 
	this->addAttribute(make_pair("dimx","int"));
	this->addAttribute(make_pair("dimy","int"));
	this->addAttribute(make_pair("dimz","int"));
}

heteroGenerator3D::~heteroGenerator3D()
{
}
void heteroGenerator3D::addSrcArray(pair<string, string> srcAndType) 
{
	_srcArrayList.insert(srcAndType);
}

void heteroGenerator3D::addDstArray(pair<string, string> dstAndType)
{
	_dstArrayList.insert(dstAndType);
}

void heteroGenerator3D::addAttribute(pair<string, string> attAndType)
{
	_attributeList.insert(attAndType);
}

void heteroGenerator3D::addSharedMem(pair<string, string> sMemAndType)
{
	_sharedMemList.insert(sMemAndType);
}

void heteroGenerator3D::addVariable(pair<string, string> varAndType)
{
	_variableList.insert(varAndType);
}

void heteroGenerator3D::addRegister(pair<string, string> regAndType)
{
	_registerList.insert(regAndType);
}

void heteroGenerator3D::generateHeaderFile(void)
{
	fs.open(strHppFile.c_str(), ios::out);
	
	fs	<<	"#ifndef _"	+ module +"_hpp\n";
	fs	<<	"#define _"	+ module +"_hpp\n";
	fs	<<	"#include <cuda.h>\n";
	fs	<<	"void " 	+ module +"(";
	for( map<string, string>::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
		fs << (*it).second << " " << (*it).first << ", ";
	for( map<string, string>::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
		fs << (*it).second << " " << (*it).first << ", ";
	for( map<string, string>::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
		fs << (*it).second << " " << (*it).first << ", ";	
	for( map<string, string>::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
		fs << (*it).second << " " << (*it).first << ", ";		
	fs 	<< "int halo = 0, cudaStream_t stream = 0);\n\n";
	fs 	<< "#endif\n";
	
	fs.close();
}

void heteroGenerator3D::generateKernelFile(void)
{
	fs.open(strCUDAFile.c_str(), ios::out);

	fs	<<	"#include \""+module+".hpp\"\n";
	fs	<<	"#include \"helper_math.h\" \n";
	fs 	<< 	endl;
	
	/// Interface of wrapper
	fs	<<	"void " 	+ module +"(";
for( map<string, string>::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( map<string, string>::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( map<string, string>::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).second << " " << (*it).first << ", ";	
for( map<string, string>::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";		
	fs 	<< "int halo, cudaStream_t stream);\n\n";

	/// Interface of kernel
	fs	<<	"__global__ \n";
	fs	<<	"void __" 	+ module +"(";
for( map<string, string>::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( map<string, string>::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( map<string, string>::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).second << " " << (*it).first << ", ";	
for( map<string, string>::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";		
	fs 	<< "int halo);\n\n";
	
	/// "Implement" the wrapper
	fs	<<	"void " 	+ module +"(";
for( map<string, string>::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( map<string, string>::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( map<string, string>::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).second << " " << (*it).first << ", ";	
for( map<string, string>::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";		
	fs 	<< "int halo, cudaStream_t stream)\n";
	fs	<< "{\n";
	// Kernel configuration
	fs 	<< 	"    dim3 blockDim(8, 8, 8);\n";
	fs 	<< 	"    dim3 gridDim("											<<	endl
		<< 	"        (dimx/blockDim.x + ((dimx%blockDim.x)?1:0))," 		<< 	endl
		<< 	"        (dimy/blockDim.y + ((dimy%blockDim.y)?1:0))," 		<< 	endl
		<< 	"        (dimz/blockDim.z + ((dimz%blockDim.z)?1:0)) );"  	<< 	endl;
	
	// Shared memory configuration
for( map<string, string>::iterator it=_sharedMemList.begin(); it!=_sharedMemList.end(); it++)
fs	<<	"    size_t " << "sharedMemSize" 
	<<	"  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(" <<	(*it).second 
	<<	");\n";
	
	// Invoke the kernel
	fs	<<	"    __"+module+"<<<gridDim, blockDim, sharedMemSize, stream>>>\n";
	fs	<<	"     (";
for( map<string, string>::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).first << ", ";
for( map<string, string>::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).first << ", ";
for( map<string, string>::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).first << ", ";
for( map<string, string>::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).first << ", ";
	fs 	<< "halo);\n";
	fs	<< "}\n\n";
	
	/// Flattern indices to 1d
	fs	<<	"#define at(x, y, z, dimx, dimy, dimz) ( clamp(z, 0, dimz-1)*dimy*dimx +       \\\n";	
	fs	<<	"                                        clamp(y, 0, dimy-1)*dimx +            \\\n";	
	fs	<<	"                                        clamp(x, 0, dimx-1) )                   \n";	
	
	/// "Implement" the kernel

	fs	<<	"__global__ \n";
	fs	<<	"void __" 	+ module +"(";
for( map<string, string>::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( map<string, string>::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( map<string, string>::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).second << " " << (*it).first << ", ";	
for( map<string, string>::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";		
	fs 	<< "int halo)\n";
	fs	<< "{\n";
	// Shared memory declaration
for( map<string, string>::iterator it=_sharedMemList.begin(); it!=_sharedMemList.end(); ++it) 
	fs 	<<  "    extern __shared__ " << (*it).second << " " << (*it).first << "[];                     										\n";
	fs	<< 	"    int  shared_index_1d, global_index_1d, index_1d;                                      										\n";
	fs	<< 	"    int3 shared_index_3d, global_index_3d, index_3d;                                      										\n";
	fs	<< 	"    // Multi batch reading here                                                           										\n";
	fs	<< 	"    int3 sharedMemDim    = make_int3(blockDim.x+2*halo,                                   										\n";
	fs	<< 	"                                     blockDim.y+2*halo,                                  										\n";
	fs	<< 	"                                     blockDim.z+2*halo);                                  										\n";
	fs	<< 	"    int  sharedMemSize   = sharedMemDim.x*sharedMemDim.y*sharedMemDim.z;                  										\n";
	fs	<< 	"    int3 blockSizeDim    = make_int3(blockDim.x+0*halo,                                   										\n";
	fs	<< 	"                                     blockDim.y+0*halo,                                   										\n";
	fs	<< 	"                                     blockDim.z+0*halo);                                  										\n";
	fs	<< 	"    int  blockSize        = blockSizeDim.x*blockSizeDim.y*blockSizeDim.z;                  									\n";
	fs	<< 	"    int  numBatches       = sharedMemSize/blockSize + ((sharedMemSize%blockSize)?1:0);     									\n";
	fs	<< 	"    for(int batch=0; batch<numBatches; batch++)                                           										\n";
	fs	<< 	"    {                                                                                     										\n";
	fs	<< 	"        shared_index_1d  =  threadIdx.z * blockDim.y * blockDim.x +                       										\n";
	fs	<< 	"                            threadIdx.y * blockDim.x +                                    										\n";
	fs	<< 	"                            threadIdx.x +                                                 										\n";
	fs	<< 	"                            blockSize*batch; //Magic is here quantm@unist.ac.kr           										\n";
	fs	<< 	"        shared_index_3d  =  make_int3((shared_index_1d % ((blockDim.x+2*halo)*(blockDim.x+2*halo))) % (blockDim.x+2*halo),		\n";
	fs	<< 	"                                      (shared_index_1d % ((blockDim.x+2*halo)*(blockDim.x+2*halo))) / (blockDim.x+2*halo),		\n";
	fs	<< 	"                                      (shared_index_1d / ((blockDim.x+2*halo)*(blockDim.x+2*halo))) );      					\n";
	fs	<< 	"        global_index_3d  =  make_int3(blockIdx.x * blockDim.x + shared_index_3d.x - halo, 										\n";
	fs	<< 	"                                      blockIdx.y * blockDim.y + shared_index_3d.y - halo, 										\n";
	fs	<< 	"                                      blockIdx.z * blockDim.z + shared_index_3d.z - halo);										\n";
	fs	<< 	"        global_index_1d  =  global_index_3d.z * dimy * dimx +                                    								\n";
	fs	<< 	"                            global_index_3d.y * dimx +                                    										\n";
	fs	<< 	"                            global_index_3d.x;                                            										\n";
	fs	<< 	"        if (shared_index_3d.z < (blockDim.z + 2*halo))                                    										\n";
	fs	<< 	"        {                                                                                 										\n";
	fs	<< 	"            if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&                      										\n";
	fs	<< 	"               global_index_3d.y >= 0 && global_index_3d.y < dimy &&                        									\n";
	fs	<< 	"               global_index_3d.x >= 0 && global_index_3d.x < dimx)                        										\n";
	fs	<< 	"            {                                                                             										\n";
	map<string, string>::iterator it1, it2; //Query the source (it2) to shared memory (it1)
	it1 = _sharedMemList.begin(); 	
	it2 = _srcArrayList.begin();
while(it1!=_sharedMemList.end() && it2!=_srcArrayList.end()) 
{
	fs  <<  "                "+it1->first+"[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = "+it2->first+"[global_index_1d];                         \n";
	++it1; ++it2;
}
	fs	<< 	"            }                                                                             						\n";						
	fs  <<  "            else                                                                          						\n";
	fs  <<  "            {                                                                             						\n";
	it1 = _sharedMemList.begin(); 	
	it2 = _srcArrayList.begin();
while(it1!=_sharedMemList.end() && it2!=_srcArrayList.end()) 
{
	fs  <<  "                "+it1->first+"[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = -1;                                                     \n";
	++it1; ++it2;
}
	fs  <<  "            }                                                                             \n";
	fs	<< 	"        }                                                                                 \n";
	fs	<< 	"        __syncthreads();                                                                  \n";
	fs	<< 	"    }                                                                                     \n";	
	fs	<<  "                                                                                          \n";	
	fs	<< 	"    // Stencil  processing here                                                           \n";
	it1 = _registerList.begin();
	it2 = _sharedMemList.begin(); 	
while(it1!=_registerList.end() && it2!=_sharedMemList.end()) 
{
	fs  <<  "    "+it1->second+" "+it1->first+" = "+it2->first+"[at(shared_index_3d.x + halo, shared_index_3d.y + halo, shared_index_3d.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];                         \n";
	++it1; ++it2;
}
	fs	<<  "	                                                                                       \n";
	fs	<< 	"    // Single pass writing here                                                           \n";
	fs	<< 	"    index_3d       =  make_int3(blockIdx.x * blockDim.x + threadIdx.x,                    \n";	
	fs	<< 	"                                blockIdx.y * blockDim.y + threadIdx.y,                    \n";	
	fs	<< 	"                                blockIdx.z * blockDim.z + threadIdx.z);                   \n";	
	fs	<< 	"    index_1d       =  index_3d.z * dimy * dimx +                                          \n";	
	fs	<< 	"                      index_3d.y * dimx +                                                 \n";	
	fs	<< 	"                      index_3d.x;                                                         \n";	
	fs	<<  "	                                                                                       \n";
	fs	<<  "    if (index_3d.z < dimz &&                                                              \n";
	fs	<<  "        index_3d.y < dimy &&                                                                \n";
	fs	<<  "        index_3d.x < dimx)                                                                \n";
	fs	<<  "    {                                                                                     \n";
	it1 = _dstArrayList.begin(); 	
	it2 = _registerList.begin();
while(it1!=_dstArrayList.end() && it2!=_registerList.end()) 
{
	fs  <<  "        "+it1->first+"[index_1d] = "+it2->first+";                                        \n";
	++it1; ++it2;
}
	
	fs	<<  "    }                                                                                     \n";
	fs	<<  "}                                                                                         \n";
	fs.close();
}


int main(int argc, char **argv)
{
	heteroGenerator3D gen("stencil_3d");
	
	gen.addSrcArray(make_pair("deviceSrc","float*"));
	gen.addDstArray(make_pair("deviceDst","float*"));
	
	gen.addRegister(make_pair("result"   ,"float"));
	gen.addSharedMem(make_pair("sharedMemSrc","float"));
	
	gen.generateHeaderFile();
	gen.generateKernelFile();
	
	return 0;
}