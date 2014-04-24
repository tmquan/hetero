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
	// void setMaxProcess(int); //should be less than or equal to maxProcesses
	void setNumProcess(int); //should be less than or equal to maxProcesses
	void printInfo(); 
	void addApplication(string); 
	void run(); 
private:
	int maxProcesses;
	int numProcesses;
	int numNodes;
	int numTasks;
	
	// For query the file
	vector<string> nodeList;
	int numSlots;
	int maxSlots; 
};

heteroSystem1D::heteroSystem1D()
{
	this->numTasks 		= 0;
	this->numNodes 		= 0;
	this->maxProcesses 	= 0;
	this->numProcesses	= -1;
	this->nodeList.resize(this->numNodes);
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
	cout << hostFile << endl;
	
	// Read the hostfile and decode to system parameters
	string line;
	
	this->numTasks 		= 0;
	this->numNodes 		= 0;
	this->maxProcesses 	= 0;
	this->numProcesses	= -1;
	
	getline(ifs,line);
	
	// First get the front end
	cout << line << endl;
	while(!ifs.eof())
	{
		getline(ifs,line);
		cout << line << endl;
		vector<string> vecLine = split(line, ' ');
		for(vector<string>::iterator i=vecLine.begin(); i!=vecLine.end(); ++i)
		{
			// cout << (*i) << endl;
			vector<string> component = split((*i), '=');
			if((*component.begin()) != "slots" && (*component.begin()) != "max_slots")
			{
				// cout << *component.begin() << endl;
				nodeList.push_back(*component.begin());
				this->numNodes++;
			}
			if((*component.begin()) == "max_slots")
			{
				// cout << *component.begin() << endl;
				// cout << *(++component.begin()) << endl;
				int slots = 0;
				// Query the slot
				stringstream ss;
				ss << *(++component.begin());
				ss >> slots;
				this->maxProcesses += slots;
			}
		}
	}

	// Indicate that the number of processes will be the maxProcesses (full GPU utilizations)
	this->numProcesses = this->maxProcesses;
	ifs.close();
}


heteroSystem1D::~heteroSystem1D()
{
}

void heteroSystem1D::printInfo()
{
	cout << "Number of maxProcesses :	" << maxProcesses << endl;
	cout << "Number of Processes    :	" << numProcesses << endl;
	cout << "Number of Nodes        :	" << numNodes << endl;
	if(numNodes>0)
		for(vector<string>::iterator it = nodeList.begin(); it!=nodeList.end(); ++it)
			cout << *(it) << endl;
}

// void heteroSystem1D::addApplication(string application)
// {
	// stringstream ss;
	// ss << "mpirun " << "-np " << numProcesses << " "
					// << application 
					// << endl;
	// // ss >> command;
	// command = ss.str(); //Convert string stream to string
	// cout << command << endl; // Debug
	// // system(command.c_str());
// }

// void heteroSystem1D::run()
// {
	// system(command.c_str());
// }
//----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	// heteroSystem1D sys;
	// sys.printInfo();
	string hostFile = argv[1];
	heteroSystem1D system(hostFile);
	system.printInfo();
	return 0;
}