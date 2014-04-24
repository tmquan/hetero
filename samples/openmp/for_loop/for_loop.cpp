#include <iostream>
#include <stdio.h>
#include <string.h>
#include <omp.h>

using namespace std;

int main (int argc, char *argv[])
{
	string mpirunCommand;
	omp_set_num_threads(8);
	int numThreads = 18;
	
	int chunk = 4;
	//----------------------------------------------------------------------------
	cout << endl << "For " << endl;
	#pragma omp parallel for     		\
				
	for(int thread=0; thread<numThreads; thread++)
	{
		int threadId = omp_get_thread_num();
		#pragma omp critical
		{
			// cout << "Thread Index      : " << threadId << endl;
			cout << threadId << " ";
		}
	}
	//----------------------------------------------------------------------------
	cout << endl << "Static " << endl;
	#pragma omp parallel for     		\
		schedule(static)     			\
		ordered     					\
				
	for(int thread=0; thread<numThreads; thread++)
	{
		int threadId = omp_get_thread_num();
		#pragma omp critical
		{
			// cout << "Thread Index      : " << threadId << endl;
			cout << threadId << " ";
		}
	}
	//----------------------------------------------------------------------------
	cout << endl << "Dynamic " << endl;
	#pragma omp parallel for     		\
		schedule(dynamic)    			\
		ordered     					\
				
	for(int thread=0; thread<numThreads; thread++)
	{
		int threadId = omp_get_thread_num();
		#pragma omp critical
		{
			// cout << "Thread Index      : " << threadId << endl;
			cout << threadId << " ";
		}
	}
	//----------------------------------------------------------------------------
	cout << endl << "Static with chunk " << endl;
	#pragma omp parallel for     		\
		schedule(static, chunk)			\
		ordered     					\
				
	for(int thread=0; thread<numThreads; thread++)
	{
		int threadId = omp_get_thread_num();
		#pragma omp critical
		{
			// cout << "Thread Index      : " << threadId << endl;
			cout << threadId << " ";
		}
	}
	//----------------------------------------------------------------------------
	cout << endl << "Dynamic with chunk" << endl;
	#pragma omp parallel for     		\
		schedule(dynamic, chunk)		\
		ordered     					\
				
	for(int thread=0; thread<numThreads; thread++)
	{
		int threadId = omp_get_thread_num();
		#pragma omp critical
		{
			// cout << "Thread Index      : " << threadId << endl;
			cout << threadId << " ";
		}
	}

	return 0;
}


