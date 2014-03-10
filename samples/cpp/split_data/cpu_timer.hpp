#ifndef __CPU_TIMER_HPP__
#define __CPU_TIMER_HPP__

#include <sys/time.h>                // for gettimeofday()

struct CpuTimer
{
	struct timeval start, stop;
	// double elapsed;

	
	CpuTimer()
	{
		// cudaEventCreate(&start);
		// cudaEventCreate(&stop);
	}

	~CpuTimer()
	{
		// cudaEventDestroy(start);
		// cudaEventDestroy(stop);
	}


	void Start()
	{
		gettimeofday(&start, NULL);
	}

	void Stop()
	{
		gettimeofday(&stop, NULL);
	}


	float Elapsed()
	{
		float elapsed = 0.0f;
		elapsed         += (stop.tv_sec - start.tv_sec) * 1000.0;      // sec to ms
		elapsed         += (stop.tv_usec - start.tv_usec) / 1000.0;   // us to ms
		
		return elapsed;
	}
};
#endif 