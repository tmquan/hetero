
#define DIMX 512
#define DIMY 512
#define DIMZ 512
#define ILP  8
#define BLKX 8
#define BLKY 8
#define BLKZ 8
// #define DIMX 512
// #define DIMY 512
// #define DIMZ 512

// #define ILP 8
//Written by professor Won-Ki Jeong
// wkjeong@unist.ac.kr

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_math.h>


#define SLICE DIMX*DIMY
#define C0 0.25
#define C1 0.5

// #define CORRECTNESS
#define checkLastError() {                                          				\
	cudaError_t error = cudaGetLastError();                               			\
	int id; 																		\
	cudaGetDevice(&id);																\
	if(error != cudaSuccess) {                                         				\
		printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",	\
			__FILE__,__LINE__, cudaGetErrorString(error), id);			      	 	\
		exit(EXIT_FAILURE);  														\
	}                                                               				\
}


__global__ void heatflow_global(float *a, float *c)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;


	int offset = DIMX*y + x;

	int idx, t;
	float center, left, right, top, bottom, front, back;

	#pragma unroll 
	for(int i=0; i<ILP; i++)
	{
		idx = SLICE*(ILP*z+i) + offset;
		
		center = a[idx];

		t = (x == 0) ? 0 : -1;
		left = a[idx + t];
	
		t = (x == DIMX-1) ? 0 : 1;
		right = a[idx + t];

		t = (y == 0) ? 0 : -DIMX;
		top = a[idx + t];

		t = (y == DIMY-1) ? 0 : DIMX;
		bottom = a[idx + t];

		t = (z == 0) ? 0 : -SLICE;
		front = a[idx + t];

		// t = (z == DIMZ-1) ? 0 : SLICE;
		t = ((ILP*z + i) == (DIMZ-1)) ? 0 : SLICE;
		back = a[idx + t];
		
		c[idx] = C0*center + C1*(left + right + top + bottom + front + back);
	}
}

__global__ void heatflow_shared(float *a, float *c)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	// __shared__ float sharedMem[][][1];
	int offset = DIMX*y + x;

	int idx, t;
	float center, left, right, top, bottom, front, back;

	#pragma unroll 
	for(int i=0; i<ILP; i++)
	{
		idx = SLICE*(ILP*z+i) + offset;
		
		center = a[idx];

		t = (x == 0) ? 0 : -1;
		left = a[idx + t];
	
		t = (x == DIMX-1) ? 0 : 1;
		right = a[idx + t];

		t = (y == 0) ? 0 : -DIMX;
		top = a[idx + t];

		t = (y == DIMY-1) ? 0 : DIMX;
		bottom = a[idx + t];

		t = (z == 0) ? 0 : -SLICE;
		front = a[idx + t];

		// t = (z == DIMZ-1) ? 0 : SLICE;
		t = ((ILP*z + i) == (DIMZ-1)) ? 0 : SLICE;
		back = a[idx + t];
		
		c[idx] = C0*center + C1*(left + right + top + bottom + front + back);
	}
}



int main()
{
	//srand ( time(NULL) );

	//int gpuid = rand() % 8;
	//printf("Assigned GPU ID: %d\n", gpuid);
	//cudaSetDevice( gpuid ); 

	// Allocate GPU memory	
	float *d_src, *d_dst, *h_dst, *h_src;
	
	printf("BLKX (%d), BLKY (%d), BLKZ (%d), ILP (%d)\n",
		BLKX, BLKY, BLKZ, ILP);

	h_src = (float*)malloc(sizeof(float)*DIMX*DIMY*DIMZ);
	h_dst = (float*)malloc(sizeof(float)*DIMX*DIMY*DIMZ);
	cudaMalloc((void**)&(d_src),sizeof(float)*DIMX*DIMY*DIMZ);
	cudaMalloc((void**)&(d_dst),sizeof(float)*DIMX*DIMY*DIMZ);
	
	for(int i=0; i<DIMX*DIMY*DIMZ; i++) h_src[i] = 7;

	cudaMemcpy(d_src, h_src, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice);
	cudaMemset(d_dst, 0, sizeof(float)*DIMX*DIMY*DIMZ);


	cudaEvent_t begin, end;

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaEventRecord(begin, 0);


	// call your kernel here
	// dim3 dimB = dim3(512,1,1); // block size
	dim3 dimB = dim3(BLKX,BLKY,BLKZ); // block size
	dim3 dimG = dim3(DIMX/dimB.x,DIMY/dimB.y,DIMZ/(dimB.z*ILP));

	// parameters for performance evaluation
	double flops, gbps, nops, nbp;
	int iter = 20;

#define HEATFLOW 
#ifdef HEATFLOW
	nbp = 8*4; // # of bytes transferred per point
	nops = 8.; // # of flops per point
	
	for(int i=0; i<iter; i++)
	{
		heatflow_global<<<dimG,dimB>>>(d_src, d_dst);
	}
#endif
	checkLastError();
	//
	cudaDeviceSynchronize();
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	float totalTime;
	cudaEventElapsedTime(&totalTime, begin, end);

	gbps = nbp*DIMX*DIMY*DIMZ/(totalTime/1000.)/(1024.*1024.*1024.)*(double)iter;
	flops = nops*DIMX*DIMY*DIMZ/(totalTime/1000.)/(1024.*1024.*1024.)*(double)iter;
	printf("Computing time : %.3f msec, Device memory bandwidth : %.3f GB/s, GFLOPS : %.3f\n", totalTime, gbps, flops);

	/// Verify the correctness
	cudaMemcpy(h_dst, d_dst, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyDeviceToHost);
	#ifdef CORRECTNESS
	#define at(x, y, z, dimx, dimy, dimz) ( clamp((int)(z), 0, dimz-1)*dimy*dimx +  \
											clamp((int)(y), 0, dimy-1)*dimx+       	\
											clamp((int)(x), 0, dimx-1) )                  
	float* h_ref = new float[DIMX*DIMY*DIMZ];
	float tmp, result;
	for(int z=0; z<DIMZ; z++)
	{
		for(int y=0; y<DIMY; y++)
		{
			for(int x=0; x<DIMX; x++)
			{
				tmp = C1 *    ( h_src[at(x+1, y+0, z+0, DIMX, DIMY, DIMZ)] +
								h_src[at(x-1, y+0, z+0, DIMX, DIMY, DIMZ)] +
								h_src[at(x+0, y+1, z+0, DIMX, DIMY, DIMZ)] +
								h_src[at(x+0, y-1, z+0, DIMX, DIMY, DIMZ)] +
								h_src[at(x+0, y+0, z+1, DIMX, DIMY, DIMZ)] +
								h_src[at(x+0, y+0, z-1, DIMX, DIMY, DIMZ)] );
				result = 	 C0*h_src[at(x+0, y+0, z+0, DIMX, DIMY, DIMZ)]  + tmp;			
				h_ref[at(x, y, z, DIMX, DIMY, DIMZ)] 	= result;
			}
		}
	}

	for(int z=0; z<DIMZ; z++)
	{
		for(int y=0; y<DIMY; y++)
		{
			for(int x=0; x<DIMX; x++)
			{
				if(h_ref[z*DIMY*DIMX+y*DIMX+x] != h_dst[z*DIMY*DIMX+y*DIMX+x])
				{
					printf("Solution does not match at x: %d, y: %d, z: %d\n", x, y, z);
					printf("h_ref (%04.4f), h_dst (%04.4f)\n", 
						h_ref[z*DIMY*DIMX+y*DIMX+x], 
						h_dst[z*DIMY*DIMX+y*DIMX+x]);
					return -1;
					// goto cleanup;
				}
			}
		}
	}
	printf("Solution is correct.\n");
	
	
	
	cudaFree( d_src );
	cudaFree( d_dst );
	#endif

	return 0;
}



