#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <sstream>

#include <stdlib.h>     // exit, EXIT_FAILURE 

using namespace std;
class heteroSystem1D
{
public:
	heteroSystem1D();
	heteroSystem1D(string);
	~heteroSystem1D();
	
	
	void loadHostFile(string);
	// void configureMaxProcesses();
	// void setMaxProcess(int); //should be less than or equal to maxClusterProcesses
	void setNumProcess(int); //should be less than or equal to maxClusterProcesses
	void setNumTasks(int); // Can be greater than maxClusterProcesses
	void printInfo(); 
	void addApplication(string); 
	void run(); 
private:
	int maxClusterProcesses;
	int numClusterProcesses;
	int numNodes;
	int numVirtualProcesses;
	
	string command; 	// For application
	string application; // For application
	
	// For query the file
	string hostFile;
	vector<string> nodeList;
	int numSlots;
	int maxSlots; 
	// int chunkDim;
	// int chunkIdx;
};

heteroSystem1D::heteroSystem1D()
{
	this->numNodes 				= 0;
	this->numVirtualProcesses 	= 0;
	this->maxClusterProcesses 	= 0;
	this->numClusterProcesses	= -1;
	this->nodeList.resize(this->numNodes);
	// this->chunkDim		= this->numClusterProcesses;
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

heteroSystem1D::heteroSystem1D(string hostFile)
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
	
	this->numNodes 				= 0;
	this->numVirtualProcesses 	= 0;
	this->maxClusterProcesses 	= 0;
	this->numClusterProcesses	= -1;
	
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
				this->maxClusterProcesses += slots;
			}
		}
	}

	// Indicate that the number of processes will be the maxClusterProcesses (full GPU utilizations)
	this->numClusterProcesses = this->maxClusterProcesses;
	// this->chunkDim	   = this->numClusterProcesses;
	// this->chunkIdx	   = 0;
	this->hostFile     = hostFile;
	ifs.close();
}


heteroSystem1D::~heteroSystem1D()
{
}

void heteroSystem1D::printInfo()
{
	cout << "Number of maxClusterProcesses :	" << maxClusterProcesses << endl;
	cout << "Number of Processes    :	" << numClusterProcesses << endl;
	cout << "Number of Nodes        :	" << numNodes << endl;
	
	if(numNodes>0)
		for(vector<string>::iterator it = nodeList.begin(); it!=nodeList.end(); ++it)
			cout << "- " << *(it) << endl;
}

void heteroSystem1D::addApplication(string application)
{
	cout << "Command line Application has been added :" << endl;
	cout << "#" << application << endl;
	this->application =  application;
}

void heteroSystem1D::setNumTasks(int numVirtualProcesses)
{
	cout << "Number of tasks :" << numVirtualProcesses << endl;
	this->numVirtualProcesses =  numVirtualProcesses;
}

void heteroSystem1D::run()
{
	// // Declare hyper rank indices
	// // int global_rank_index = 0;
	// // int local_rank_index = 0;
	// int rankIdx  = 0;
	// int chunkDim = this->numClusterProcesses;
	
	// //Round up the chunk Dimension
	// int hyperDim = (numVirtualProcesses/chunkDim + ((numVirtualProcesses%chunkDim)?1:0));
	// // cout << "Hyper Dimension : " << hyperDim << endl;
	
	// // int thisNumTasks=0;
	// int taskIdx = 0;
	// int chunkIdx = 0;
	// //
	// for(chunkIdx=0; chunkIdx<hyperDim; chunkIdx++)
	// {
		// for(rankIdx=0; rankIdx<chunkDim; rankIdx++)
		// {
			// //chunkIdx*chunkDim will be the offset to pass into the application
			// taskIdx = chunkIdx*chunkDim + rankIdx;
			// if(taskIdx==(numVirtualProcesses-1)) break;
		// }
		// // cout << "Task Index " << taskIdx << endl;
		// // Count how many processes we need to launch
		// // thisNumTasks = ((taskIdx+1)%chunkDim)?((taskIdx+1)%chunkDim):chunkDim;
		// // cout << thisNumTasks << endl;
		// numClusterProcesses = ((taskIdx+1)%chunkDim)?((taskIdx+1)%chunkDim):chunkDim;
		// //////////////////////////////////////////////////////////
		// stringstream ss;
		// ss << "mpirun ";
		// // ss << "  /cm/shared/custom/apps/openmpi-gpudirect/gcc/64/1.7.2/bin/mpirun ";
		// ss << " --np " << numClusterProcesses;
		// // ss << " --np " << thisNumTasks;
		// ss << " --host ";
		// for(vector<string>::iterator it = nodeList.begin(); it!=nodeList.end(); ++it)
		// {
			// ss <<  *(it);
			// if(it!=nodeList.end()-1)
			// ss << ",";
		// }
		// // ss << " --byslot ";
		// // ss << " --machinefile " << hostFile;
		// ss << " " << application;
		// ss << " |sort";
		
		
		// ss >> command;
		// command = ss.str(); //Convert string stream to string
		// cout << "MPI Command Line :" << endl;
		// cout << "#" << command << endl; // Debug
		// // // system(command.c_str());
		// system(command.c_str());
	// }
}
//----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	// heteroSystem1D sys;
	// sys.printInfo();
	string hostFile = argv[1];
	heteroSystem1D system(hostFile);
	system.printInfo();
	// system.setNumTasks(18);
	// // system.addApplication("echo $(hostname)");
	// system.addApplication("/home/tmquan/hetero/build/bin/mpi-hello_world");
	// system.run();
	
	system.setNumTasks(13);
	// system.addApplication("echo $(hostname)");
	system.addApplication("/home/tmquan/hetero/build/bin/mpi-hello_world");
	system.run();
	return 0;
}