#include <stdio.h>
#include <omp.h>
int main (int argc, char *argv[])
{
	int max_threads = omp_get_max_threads();
	#pragma omp parallel num_threads(max_threads)
	{
		int thread = omp_get_thread_num();
		#pragma omp critical
		{
			printf("Hello World from thread %02d out of %02d\n", thread, max_threads);
		}
	}

	return 0;
}


