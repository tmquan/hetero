#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <cuda.h>
#include <helper_math.h>
#include <stdlib.h>     // exit, EXIT_FAILURE 

using namespace std;
class heteroSystem
{
public:
	heteroSystem();
	heteroSystem(string);
	~heteroSystem();
	
	
	void loadHostFile(string);
	// void configureMaxProcesses();
	// void setMaxProcess(int); //should be less than or equal to clusterSize
	
	int getNumRun(); // get the number of query
	void setNumProcesses(int); //should be less than or equal to clusterSize
	void setVirtualSize(int); // Can be greater than clusterSize
	void printInfo(); 
	void addApplication(string); 
	void run(); 
private:
	int clusterSize;
	int numProcesses;
	int numNodes;
	int virtualSize;
	int numProcessesWillBeLaunched;
	
	string command; 	// For application
	string application; // For application
	
	// For query the file
	string hostFile;
	vector<string> nodeList;
	int numSlots;
	int maxSlots; 
	// int chunkDim;
	// int chunkIdx;
	int numRun;
	int execId;
};

heteroSystem::heteroSystem()
{
	this->numNodes 		= 0;
	this->numRun 		= 0;
	this->execId 		= 0;
	this->virtualSize 	= 0;
	this->clusterSize 	= 0;
	this->numProcesses	= -1;
	this->nodeList.resize(this->numNodes);
	// this->chunkDim		= this->numProcesses;
	// this->chunkIdx		= 0;
}



std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

heteroSystem::heteroSystem(string hostFile)
{
	ifstream ifs;
	ifs.open(hostFile.c_str());
	if(!ifs.is_open())
	{
		cout << "Cannot open the hostfile: " << hostFile << endl;
		return;
	}
	cout << "Location of Host file  :	" << hostFile << endl;
	
	// Read the hostfile and decode to system parameters
	string line;
	
	this->numNodes 		= 0;
	this->virtualSize 	= 0;
	this->clusterSize 	= 0;
	this->numProcesses	= -1;
	
	cout << "Content of Host file   :	"  << endl;
	getline(ifs,line);
	
	// First get the front end
	cout << line << endl;
	while(!ifs.eof())
	{
		getline(ifs,line);
		cout << line << endl;
		vector<string> lines = split(line, ' ');
		for(vector<string>::iterator i=lines.begin(); i!=lines.end(); ++i)
		{
			// cout << (*i) << endl;
			vector<string> component = split((*i), '=');
			if((*component.begin()) != "slots" && (*component.begin()) != "max_slots")
			{
				// cout << *component.begin() << endl;
				vector<string> names = split(*component.begin(), '.');
				// nodeList.push_back(*component.begin());
				nodeList.push_back(*names.begin());
				this->numNodes++;
			}
			if((*component.begin()) == "max_slots")
			{
				// cout << *component.begin() << endl;
				// cout << *(++component.begin()) << endl;
				int slots = 0;
				// Query the slot
				stringstream ss;
				// ss << *(component.end());
				ss << *(++component.begin());
				ss >> slots;
				this->clusterSize += slots;
			}
		}
	}

	// Indicate that the number of processes will be the clusterSize (full GPU utilizations)
	this->numProcesses = this->clusterSize;
	this->hostFile     = hostFile;
	ifs.close();
	this->numRun 		= 0;
	this->execId 		= 0;
}


heteroSystem::~heteroSystem()
{
}

void heteroSystem::printInfo()
{
	cout << "Size of cluster        :	" << clusterSize << endl;
	cout << "Number of Processes    :	" << numProcesses << endl;
	cout << "Number of Nodes        :	" << numNodes << endl;
	
	if(numNodes>0)
		for(vector<string>::iterator it = nodeList.begin(); it!=nodeList.end(); ++it)
			cout << "- " << *(it) << endl;
}

void heteroSystem::addApplication(string application)
{
	cout << "Command line Application has been added :" << endl;
	cout << "#" << application << endl;
	this->application =  application;
}

void heteroSystem::setNumProcesses(int numProcesses)
{
	cout << "numProcesses :" << numProcesses << endl;
	if(numProcesses > this->clusterSize)
	{
		printf("--Warning: setting the number of processes (%d) greater than clusterSize (%d)\n", numProcesses, this->clusterSize);
		printf("--Demoted numProcesses to clusterSize (%d)\n", this->clusterSize);
		numProcesses = this->clusterSize;
	}
	this->numProcesses =  numProcesses;
	if(this->numProcesses != 1)
		this->numRun = (this->virtualSize / this->numProcesses) + ((this->virtualSize % this->numProcesses)?1:0);
	else
		this->numRun = (this->virtualSize / this->numProcesses);
}


int heteroSystem::getNumRun()
{
	cout << "num Run :" << this->numRun << endl;
	return this->numRun;
}

void heteroSystem::setVirtualSize(int virtualSize)
{
	cout << "Virtual Size :" << virtualSize << endl;
	this->virtualSize =  virtualSize;
	// if(this->numProcesses != 1)
		// this->numRun = (this->virtualSize / this->numProcesses) + ((this->virtualSize % this->numProcesses)?1:0);
	// else
		// this->numRun = (this->virtualSize / this->numProcesses);		
}


void heteroSystem::run()
{	
	for(execId=0; execId<numRun; execId++)
	{
		cout << "----------------------------------------------------------" << endl;
		// Determine the number of processes will be launched
		int numProcessesWillBeLaunched = 0;
		if(execId<(numRun-1))
			numProcessesWillBeLaunched = ((numRun>1)?(numProcesses):(virtualSize % numProcesses));
		else if(execId==(numRun-1)) // Index from 0 to numRun-1
			numProcessesWillBeLaunched = ((virtualSize % numProcesses)?(virtualSize % numProcesses):(numProcesses));
			
		//----------------------------------------------------------	
		// Construct the MPI command
		stringstream ss;
		ss << "mpirun ";
		// ss << "  /cm/shared/custom/apps/openmpi-gpudirect/gcc/64/1.7.2/bin/mpirun ";
		ss << " --np " << numProcessesWillBeLaunched;
		ss << " --host ";
		for(vector<string>::iterator it = nodeList.begin(); it!=nodeList.end(); ++it)
		{
			ss <<  *(it);
			if(it!=nodeList.end()-1)
			ss << ",";
		}
		// ss << " --bynode ";
		// ss << " --machinefile " << hostFile;
		ss << " " << application;
		ss << " --execId=" << execId;
		ss << " --maxProcs=" << numProcesses;
		ss << " | sort" << endl;
		
		
		ss >> command;
		command = ss.str(); //Convert string stream to string
		cout << "MPI Command Line :" << endl;
		cout << "#" << command << endl; // Debug
		system(command.c_str());
	}
	//----------------------------------------------------------------------------	
}
//----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	string hostFile = argv[1];
	heteroSystem system(hostFile);
	system.printInfo();
	

	// int dimx = 4455;
	// int dimy = 3408;
	// int dimz = 64;
	// // int3 processDim		{1485, 1136, 128};
	// int3 processDim		{dimx/3, dimy/3, dimz/2};
	
	// // int dimx = 500;
	// // int dimy = 450;
	// // int dimz = 400;
	// // int3 processDim		{250, 225,  200};
	
	// // int3 processDim		{256, 256,  256};

	// int3 virtualDim    	{(dimx/processDim.x + ((dimx%processDim.x)?1:0)),
						 // (dimy/processDim.y + ((dimy%processDim.y)?1:0)),
						 // (dimz/processDim.z + ((dimz%processDim.z)?1:0))};
						 
	// int virtualSize = virtualDim.x*virtualDim.y*virtualDim.z;
	// int3 haloDim {12, 12, 12};
	// // int3 haloDim {0, 0, 0};
	
	// system.setVirtualSize(virtualSize);
	// system.setNumProcesses(8);
	
	// system.getNumRun();

	
	
	// stringstream ssApp;	
	
	// Median
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_median " 
		  // << " --dimx=" << dimx
		  // << " --dimy=" << dimy
		  // << " --dimz=" << dimz
		  // << " --virtualDimx=" << virtualDim.x
		  // << " --virtualDimy=" << virtualDim.y
		  // << " --virtualDimz=" << virtualDim.z
		  // << " --processDimx=" << processDim.x
		  // << " --processDimy=" << processDim.y
		  // << " --processDimz=" << processDim.z
		  // << " --haloDimx=" << haloDim.x
		  // << " --haloDimy=" << haloDim.y
		  // << " --haloDimz=" << haloDim.z
		  // // << " --srcFile=" << "../../../../data/em_4455x3408x512.raw" 
		  // << " --srcFile=" << "../../../../data/em_500x450x400.raw" 
		  // << " --dstFile=" << "../../../../data/median_500x450x400.raw" 
		  // ;

	// system.addApplication(ssApp.str());
	// system.run();
	
	// Median
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_stddev " 
		  // << " --dimx=" << dimx
		  // << " --dimy=" << dimy
		  // << " --dimz=" << dimz
		  // << " --virtualDimx=" << virtualDim.x
		  // << " --virtualDimy=" << virtualDim.y
		  // << " --virtualDimz=" << virtualDim.z
		  // << " --processDimx=" << processDim.x
		  // << " --processDimy=" << processDim.y
		  // << " --processDimz=" << processDim.z
		  // << " --haloDimx=" << haloDim.x
		  // << " --haloDimy=" << haloDim.y
		  // << " --haloDimz=" << haloDim.z
		  // // << " --srcFile=" << "../../../../data/em_4455x3408x512.raw" 
		  // << " --srcFile=" << "../../../../data/median_500x450x400.raw" 
		  // // << " --srcFile=" << "../../../../data/em_500x450x400.raw" 
		  // // << " --srcFile=" << "../../../../data/ones_500x450x400.raw" 
		  // << " --dstFile=" << "../../../../data/stddev_500x450x400.raw" 
		  // ;
	// // Bilateral
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_bilateral " 
	  // << " --dimx=" << dimx
	  // << " --dimy=" << dimy
	  // << " --dimz=" << dimz
	  // << " --virtualDimx=" << virtualDim.x
	  // << " --virtualDimy=" << virtualDim.y
	  // << " --virtualDimz=" << virtualDim.z
	  // << " --processDimx=" << processDim.x
	  // << " --processDimy=" << processDim.y
	  // << " --processDimz=" << processDim.z
	  // << " --haloDimx=" << haloDim.x
	  // << " --haloDimy=" << haloDim.y
	  // << " --haloDimz=" << haloDim.z 
	  // << " --srcFile=" << "../../../../data/stddev_500x450x400.raw" 
	  // << " --dstFile=" << "../../../../data/bilateral_500x450x400.raw" 
	  // ;
	
	// system.addApplication(ssApp.str());
	// system.run();
	
	
	// // Minimum
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_minimum " 
	  // << " --dimx=" << dimx
	  // << " --dimy=" << dimy
	  // << " --dimz=" << dimz
	  // << " --virtualDimx=" << virtualDim.x
	  // << " --virtualDimy=" << virtualDim.y
	  // << " --virtualDimz=" << virtualDim.z
	  // << " --processDimx=" << processDim.x
	  // << " --processDimy=" << processDim.y
	  // << " --processDimz=" << processDim.z
	  // << " --haloDimx=" << haloDim.x
	  // << " --haloDimy=" << haloDim.y
	  // << " --haloDimz=" << haloDim.z 
	  // << " --srcFile=" << "../../../../data/bilateral_500x450x400.raw" 
	  // << " --dstFile=" << "../../../../data/minimum_500x450x400.raw" 
	  // ;
	
	// system.addApplication(ssApp.str());
	// system.run();
	
	
	// // Threshold
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_threshold " 
	  // << " --dimx=" << dimx
	  // << " --dimy=" << dimy
	  // << " --dimz=" << dimz
	  // << " --virtualDimx=" << virtualDim.x
	  // << " --virtualDimy=" << virtualDim.y
	  // << " --virtualDimz=" << virtualDim.z
	  // << " --processDimx=" << processDim.x
	  // << " --processDimy=" << processDim.y
	  // << " --processDimz=" << processDim.z
	  // << " --haloDimx=" << haloDim.x
	  // << " --haloDimy=" << haloDim.y
	  // << " --haloDimz=" << haloDim.z 
	  // << " --srcFile=" << "../../../../data/minimum_500x450x400.raw" 
	  // << " --dstFile=" << "../../../../data/threshold_500x450x400.raw" 
	  // ;
	
	// system.addApplication(ssApp.str());
	// system.run();
	
	
	// /// 29 GB here
	// int dimx = 4455;
	// int dimy = 3408;
	// int dimz = 512;
	
	// int3 processDim		{dimx/3, dimy/3, dimz/8};
	

	// int3 virtualDim    	{(dimx/processDim.x + ((dimx%processDim.x)?1:0)),
						 // (dimy/processDim.y + ((dimy%processDim.y)?1:0)),
						 // (dimz/processDim.z + ((dimz%processDim.z)?1:0))};
						 
	// int virtualSize = virtualDim.x*virtualDim.y*virtualDim.z;
	// int3 haloDim {12, 12, 12};
	// // int3 haloDim {0, 0, 0};
	
	// system.setVirtualSize(virtualSize);
	// system.setNumProcesses(8);
	
	// system.getNumRun();

	
	
	// stringstream ssApp;	
	
	
	// // Median
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_median " 
		  // << " --dimx=" << dimx
		  // << " --dimy=" << dimy
		  // << " --dimz=" << dimz
		  // << " --virtualDimx=" << virtualDim.x
		  // << " --virtualDimy=" << virtualDim.y
		  // << " --virtualDimz=" << virtualDim.z
		  // << " --processDimx=" << processDim.x
		  // << " --processDimy=" << processDim.y
		  // << " --processDimz=" << processDim.z
		  // << " --haloDimx=" << haloDim.x
		  // << " --haloDimy=" << haloDim.y
		  // << " --haloDimz=" << haloDim.z
		  // << " --srcFile=" << "../../../../data/em_4455x3408x512.raw" 
		  // << " --dstFile=" << "../../../../data/median_4455x3408x512.raw" 
		  // ;

	// system.addApplication(ssApp.str());
	// system.run();
	
	// // Stddev
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_stddev " 
		  // << " --dimx=" << dimx
		  // << " --dimy=" << dimy
		  // << " --dimz=" << dimz
		  // << " --virtualDimx=" << virtualDim.x
		  // << " --virtualDimy=" << virtualDim.y
		  // << " --virtualDimz=" << virtualDim.z
		  // << " --processDimx=" << processDim.x
		  // << " --processDimy=" << processDim.y
		  // << " --processDimz=" << processDim.z
		  // << " --haloDimx=" << haloDim.x
		  // << " --haloDimy=" << haloDim.y
		  // << " --haloDimz=" << haloDim.z
		  // << " --srcFile=" << "../../../../data/median_4455x3408x512.raw" 
		  // << " --dstFile=" << "../../../../data/stddev_4455x3408x512.raw" 
		  // ;
	// system.addApplication(ssApp.str());
	// system.run();
	
	// // Bilateral
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_bilateral " 
	  // << " --dimx=" << dimx
	  // << " --dimy=" << dimy
	  // << " --dimz=" << dimz
	  // << " --virtualDimx=" << virtualDim.x
	  // << " --virtualDimy=" << virtualDim.y
	  // << " --virtualDimz=" << virtualDim.z
	  // << " --processDimx=" << processDim.x
	  // << " --processDimy=" << processDim.y
	  // << " --processDimz=" << processDim.z
	  // << " --haloDimx=" << haloDim.x
	  // << " --haloDimy=" << haloDim.y
	  // << " --haloDimz=" << haloDim.z 
	  // << " --srcFile=" << "../../../../data/stddev_4455x3408x512.raw" 
	  // << " --dstFile=" << "../../../../data/bilateral_4455x3408x512.raw" 
	  // ;
	
	// system.addApplication(ssApp.str());
	// system.run();
	
	
	// // Minimum
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_minimum " 
	  // << " --dimx=" << dimx
	  // << " --dimy=" << dimy
	  // << " --dimz=" << dimz
	  // << " --virtualDimx=" << virtualDim.x
	  // << " --virtualDimy=" << virtualDim.y
	  // << " --virtualDimz=" << virtualDim.z
	  // << " --processDimx=" << processDim.x
	  // << " --processDimy=" << processDim.y
	  // << " --processDimz=" << processDim.z
	  // << " --haloDimx=" << haloDim.x
	  // << " --haloDimy=" << haloDim.y
	  // << " --haloDimz=" << haloDim.z 
	  // << " --srcFile=" << "../../../../data/bilateral_4455x3408x512.raw" 
	  // << " --dstFile=" << "../../../../data/minimum_4455x3408x512.raw" 
	  // ;
	
	// system.addApplication(ssApp.str());
	// system.run();
	
	
	// // Threshold
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_threshold " 
	  // << " --dimx=" << dimx
	  // << " --dimy=" << dimy
	  // << " --dimz=" << dimz
	  // << " --virtualDimx=" << virtualDim.x
	  // << " --virtualDimy=" << virtualDim.y
	  // << " --virtualDimz=" << virtualDim.z
	  // << " --processDimx=" << processDim.x
	  // << " --processDimy=" << processDim.y
	  // << " --processDimz=" << processDim.z
	  // << " --haloDimx=" << haloDim.x
	  // << " --haloDimy=" << haloDim.y
	  // << " --haloDimz=" << haloDim.z 
	  // << " --srcFile=" << "../../../../data/minimum_4455x3408x512.raw" 
	  // << " --dstFile=" << "../../../../data/threshold_4455x3408x512.raw" 
	  // ;
	
	// system.addApplication(ssApp.str());
	// system.run();
	
	
	// 29 GB, 1 time launched here
	int dimx = 4455;
	int dimy = 3408;
	int dimz = 512;
	
	// int3 processDim		{dimx/3, dimy/3, dimz/8};
	int3 processDim		{dimx/1, dimy/1, 1};
	int3 virtualDim    	{(dimx/processDim.x + ((dimx%processDim.x)?1:0)),
						 (dimy/processDim.y + ((dimy%processDim.y)?1:0)),
						 (dimz/processDim.z + ((dimz%processDim.z)?1:0))};
						 
	int virtualSize = virtualDim.x*virtualDim.y*virtualDim.z;
	int3 haloDim {12, 12, 12};
	
	system.setVirtualSize(virtualSize);
	system.setNumProcesses(8); 
	system.getNumRun();	
	stringstream ssApp;	
	
	
	// Segment
	ssApp.str("");
	ssApp << "../../../../bin/hybrid-app_segment " 
		  << " --dimx=" << dimx
		  << " --dimy=" << dimy
		  << " --dimz=" << dimz
		  << " --virtualDimx=" << virtualDim.x
		  << " --virtualDimy=" << virtualDim.y
		  << " --virtualDimz=" << virtualDim.z
		  << " --processDimx=" << processDim.x
		  << " --processDimy=" << processDim.y
		  << " --processDimz=" << processDim.z
		  << " --haloDimx=" << haloDim.x
		  << " --haloDimy=" << haloDim.y
		  << " --haloDimz=" << haloDim.z
		  << " --srcFile=" << "../../../../data/em_4455x3408x512.raw" 
		  << " --dstFile=" << "../../../../data/segment_4455x3408x512.raw" 
		  ;
	system.addApplication(ssApp.str());
	system.run();

	
	
	// // 936 GB, 1 time launched here
	// int dimx = 21494;
	// int dimy = 25790;
	// // int dimz = 1850;
	// int dimz = 36;
	
	// int3 processDim		{1000, 1000, 36};
	// int3 virtualDim    	{(dimx/processDim.x + ((dimx%processDim.x)?1:0)),
						 // (dimy/processDim.y + ((dimy%processDim.y)?1:0)),
						 // (dimz/processDim.z + ((dimz%processDim.z)?1:0))};
						 
	// int virtualSize = virtualDim.x*virtualDim.y*virtualDim.z;
	// int3 haloDim {12, 12, 12};
	
	// system.setVirtualSize(virtualSize);
	// system.setNumProcesses(8);
	// system.getNumRun();	
	// stringstream ssApp;	
	
	
	// // Segment
	// ssApp.str("");
	// ssApp << "../../../../bin/hybrid-app_segment " 
		  // << " --dimx=" << dimx
		  // << " --dimy=" << dimy
		  // << " --dimz=" << dimz
		  // << " --virtualDimx=" << virtualDim.x
		  // << " --virtualDimy=" << virtualDim.y
		  // << " --virtualDimz=" << virtualDim.z
		  // << " --processDimx=" << processDim.x
		  // << " --processDimy=" << processDim.y
		  // << " --processDimz=" << processDim.z
		  // << " --haloDimx=" << haloDim.x
		  // << " --haloDimy=" << haloDim.y
		  // << " --haloDimz=" << haloDim.z
		  // << " --srcFile=" << "../../../../data/em_21494x25790x1850.raw" 
		  // << " --dstFile=" << "../../../../data/segment_21494x25790x1850.raw" 
		  // ;
	// system.addApplication(ssApp.str());
	// system.run();
	return 0;
}