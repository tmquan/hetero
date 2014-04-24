#include <iostream>
#include <stdio.h>
#include <string.h>
#include <omp.h>

using namespace std;

int main (int argc, char *argv[])
{
	string mpirunCommand;
	omp_set_num_threads(8);
	int numHyperProcesses = 16;
	
	int chunk = 4;
	
	cout << "Static" << endl;
	#pragma omp parallel for     		\
		schedule(static,chunk)     		\
		ordered     					\
				
	for(int process=0; process<numHyperProcesses; process++)
	{
		int threadId = omp_get_thread_num();
		#pragma omp critical
		{
			cout << "Process Index      : " << threadId << endl;
		}
	}
	
	cout << "Dynamic " << endl;
	#pragma omp parallel for     		\
		schedule(dynamic,chunk)    		\
		ordered     					\
				
	for(int process=0; process<numHyperProcesses; process++)
	{
		int threadId = omp_get_thread_num();
		#pragma omp critical
		{
			cout << "Process Index      : " << threadId << endl;
		}
	}

	return 0;
}


