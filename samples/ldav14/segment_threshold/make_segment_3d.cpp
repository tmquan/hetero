#include <iostream>
#include <fstream>
#include <vector>
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
	vector< pair<string, string> > 	_srcArrayList; //name and type
	vector< pair<string, string> > 	_dstArrayList;
	vector< pair<string, string> > 	_attributeList;
	vector< pair<string, string> > 	_sharedMemList;
	vector< pair<string, string> > 	_variableList;
	vector< pair<string, string> > 	_registerList;
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
	_srcArrayList.push_back(srcAndType);
}

void heteroGenerator3D::addDstArray(pair<string, string> dstAndType)
{
	_dstArrayList.push_back(dstAndType);
}

void heteroGenerator3D::addAttribute(pair<string, string> attAndType)
{
	_attributeList.push_back(attAndType);
}

void heteroGenerator3D::addSharedMem(pair<string, string> sMemAndType)
{
	_sharedMemList.push_back(sMemAndType);
}

void heteroGenerator3D::addVariable(pair<string, string> varAndType)
{
	_variableList.push_back(varAndType);
}

void heteroGenerator3D::addRegister(pair<string, string> regAndType)
{
	_registerList.push_back(regAndType);
}

void heteroGenerator3D::generateHeaderFile(void)
{
	fs.open(strHppFile.c_str(), ios::out);
	
	fs	<<	"#ifndef _"	+ module +"_hpp\n";
	fs	<<	"#define _"	+ module +"_hpp\n";
	fs	<<	"#include <cuda.h>\n";
	fs	<<	"void " 	+ module +"(";
	for( vector< pair<string, string> > ::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
		fs << (*it).second << " " << (*it).first << ", ";
	for( vector< pair<string, string> > ::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
		fs << (*it).second << " " << (*it).first << ", ";
	for( vector< pair<string, string> > ::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
		fs << (*it).second << " " << (*it).first << ", ";	
	for( vector< pair<string, string> > ::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
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
for( vector< pair<string, string> > ::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).second << " " << (*it).first << ", ";	
for( vector< pair<string, string> > ::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";		
	fs 	<< "int halo, cudaStream_t stream);\n\n";

	/// Interface of kernel
	fs	<<	"__global__ \n";
	fs	<<	"void __" 	+ module +"(";
for( vector< pair<string, string> > ::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).second << " " << (*it).first << ", ";	
for( vector< pair<string, string> > ::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";		
	fs 	<< "int halo);\n\n";
	
	/// "Implement" the wrapper
	fs	<<	"void " 	+ module +"(";
for( vector< pair<string, string> > ::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).second << " " << (*it).first << ", ";	
for( vector< pair<string, string> > ::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
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
for( vector< pair<string, string> > ::iterator it=_sharedMemList.begin(); it!=_sharedMemList.end(); it++)
fs	<<	"    size_t " << "sharedMemSize" 
	<<	"  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(" <<	(*it).second 
	<<	");\n";
	
	// Invoke the kernel
	fs	<<	"    __"+module+"<<<gridDim, blockDim, sharedMemSize, stream>>>\n";
	fs	<<	"     (";
for( vector< pair<string, string> > ::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).first << ", ";
	fs 	<< "halo);\n";
	fs	<< "}\n\n";
	
	/// Flattern indices to 1d
	fs	<<	"inline __device__ __host__ int clamp_mirror(int f, int a, int b)      				\n";	
	fs	<<	"{      																			\n";	
	fs	<<	"    if(f<a) return (a+(a-f));														\n";	
	fs	<<	"    if(f>b) return (b-(f-b));														\n";	
	fs	<<	"    return f;																		\n";	
	fs	<<	"}       																			\n";	
	
	
	fs	<<	"#define at(x, y, z, dimx, dimy, dimz) ( clamp_mirror((int)z, 0, dimz-1)*dimy*dimx +       \\\n";	
	fs	<<	"                                        clamp_mirror((int)y, 0, dimy-1)*dimx +            \\\n";	
	fs	<<	"                                        clamp_mirror((int)x, 0, dimx-1) )                   \n";	
	
	/// "Implement" the kernel

	fs	<<	"__global__ \n";
	fs	<<	"void __" 	+ module +"(";
for( vector< pair<string, string> > ::iterator it=_srcArrayList.begin(); it!=_srcArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_dstArrayList.begin(); it!=_dstArrayList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";
for( vector< pair<string, string> > ::iterator it=_variableList.begin(); it!=_variableList.end(); ++it)
	fs 	<< (*it).second << " " << (*it).first << ", ";	
for( vector< pair<string, string> > ::iterator it=_attributeList.begin(); it!=_attributeList.end(); it++)
	fs 	<< (*it).second << " " << (*it).first << ", ";		
	fs 	<< "int halo)\n";
	fs	<< "{\n";
	// Shared memory declaration
for( vector< pair<string, string> > ::iterator it=_sharedMemList.begin(); it!=_sharedMemList.end(); ++it) 
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
	fs	<< 	"        shared_index_3d  =  make_int3((shared_index_1d % ((blockDim.y+2*halo)*(blockDim.x+2*halo))) % (blockDim.x+2*halo),		\n";
	fs	<< 	"                                      (shared_index_1d % ((blockDim.y+2*halo)*(blockDim.x+2*halo))) / (blockDim.x+2*halo),		\n";
	fs	<< 	"                                      (shared_index_1d / ((blockDim.y+2*halo)*(blockDim.x+2*halo))) );      					\n";
	fs	<< 	"       global_index_3d  =  make_int3(clamp_mirror(blockIdx.x * blockDim.x + shared_index_3d.x - halo, 0, dimx-1),				\n";
	fs	<< 	"                                     clamp_mirror(blockIdx.y * blockDim.y + shared_index_3d.y - halo, 0, dimy-1), 				\n";
	fs	<< 	"                                     clamp_mirror(blockIdx.z * blockDim.z + shared_index_3d.z - halo, 0, dimz-1) );			\n";
	fs	<< 	"        global_index_1d  =  global_index_3d.z * dimy * dimx +                                    								\n";
	fs	<< 	"                            global_index_3d.y * dimx +                                    										\n";
	fs	<< 	"                            global_index_3d.x;                                            										\n";
	fs	<< 	"        if (shared_index_3d.z < (blockDim.z + 2*halo))                                    										\n";
	fs	<< 	"        {                                                                                 										\n";
	fs	<< 	"            if(global_index_3d.z >= 0 && global_index_3d.z < dimz &&                      										\n";
	fs	<< 	"               global_index_3d.y >= 0 && global_index_3d.y < dimy &&                        									\n";
	fs	<< 	"               global_index_3d.x >= 0 && global_index_3d.x < dimx)                        										\n";
	fs	<< 	"            {                                                                             										\n";
	vector< pair<string, string> > ::iterator it1, it2; //Query the source (it2) to shared memory (it1)
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
	fs  <<  "    "+it1->second+" "+it1->first+" = "+it2->first+"[at(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];                         \n";
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
	fs	<<  "        index_3d.y < dimy &&                                                              \n";
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
	//---------------------------------------------------------------
	heteroGenerator3D median("median_3d");
	
	median.addSrcArray(make_pair("deviceSrc","float*"));
	median.addDstArray(make_pair("deviceDst","float*"));
	median.addAttribute(make_pair("radius","int"));
	
	median.addRegister(make_pair("result"   ,"float"));
	median.addSharedMem(make_pair("sharedMemSrc","float"));
	
	median.generateHeaderFile();
	median.generateKernelFile();
	
	//---------------------------------------------------------------
	heteroGenerator3D stddev("stddev_3d");
	
	stddev.addSrcArray(make_pair("deviceSrc","float*"));
	stddev.addDstArray(make_pair("deviceDst","float*"));
	stddev.addAttribute(make_pair("radius","int"));
	
	stddev.addRegister(make_pair("result"   ,"float"));
	stddev.addSharedMem(make_pair("sharedMemSrc","float"));
	
	stddev.generateHeaderFile();
	stddev.generateKernelFile();
	
	//---------------------------------------------------------------
	heteroGenerator3D bilateral("bilateral_3d");
	
	bilateral.addSrcArray(make_pair("deviceSrc","float*"));
	bilateral.addDstArray(make_pair("deviceDst","float*"));
	bilateral.addAttribute(make_pair("imageDensity","float"));
	bilateral.addAttribute(make_pair("colorDensity","float"));
	bilateral.addAttribute(make_pair("radius","int"));
	
	bilateral.addRegister(make_pair("result"   ,"float"));
	bilateral.addSharedMem(make_pair("sharedMemSrc","float"));
	
	bilateral.generateHeaderFile();
	bilateral.generateKernelFile();
	
	//---------------------------------------------------------------
	heteroGenerator3D minimum("minimum_3d");
	
	minimum.addSrcArray(make_pair("deviceSrc","float*"));
	minimum.addDstArray(make_pair("deviceDst","float*"));
	minimum.addAttribute(make_pair("radius","int"));
	
	minimum.addRegister(make_pair("result"   ,"float"));
	minimum.addSharedMem(make_pair("sharedMemSrc","float"));
	
	minimum.generateHeaderFile();
	minimum.generateKernelFile();
	
	//---------------------------------------------------------------
	heteroGenerator3D threshold("threshold_3d");
	
	threshold.addSrcArray(make_pair("deviceSrc","float*"));
	threshold.addDstArray(make_pair("deviceDst","float*"));
	threshold.addAttribute(make_pair("radius","int"));
	
	threshold.addRegister(make_pair("result"   ,"float"));
	threshold.addSharedMem(make_pair("sharedMemSrc","float"));
	
	threshold.generateHeaderFile();
	threshold.generateKernelFile();
	return 0;
}