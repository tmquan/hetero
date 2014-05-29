#define DIMX 	512
#define DIMY 	512
#define DIMZ 	512
#define SLICE 	DIMX*DIMY
#define ILP 	8
#define BLKX 	512
#define BLKY 	1
#define BLKZ 	(8/ILP)


#define C0 0.25
#define C1 0.5

//Written by professor Won-Ki Jeong
// wkjeong@unist.ac.kr
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
//#include <helper_math.h>
//#include <cutil_inline.h>






int clamp(int a, int b, int c)
{
   int ret = a;
   if(a < b) ret = b;
   if(a > c) ret = c;
   return ret;
}

#define at(x, y, z, DIMX, DIMY, DIMZ) ( clamp((int)(z), 0, DIMZ-1)*DIMY*DIMX +       \
                                        clamp((int)(y), 0, DIMY-1)*DIMX +            \
                                        clamp((int)(x), 0, DIMX-1) )                   


__global__ void heatflow(float *a, float *c)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;


	int offset = DIMX*y + x;

	int idx, t, nz;
	float center, left, right, top, bottom, front, back;

#pragma unroll 
	for(int i=0; i<ILP; i++)
	{
		//idx += SLICE*i;

		nz = ILP*z+i;
		idx = SLICE*(nz) + offset;
		
		center = a[idx];

		t = (x == 0) ? 0 : -1;
		left = a[idx + t];
	
		t = (x == (DIMX-1)) ? 0 : 1;
		right = a[idx + t];

		t = (y == 0) ? 0 : -DIMX;
		top = a[idx + t];

		t = (y == (DIMY-1)) ? 0 : DIMX;
		bottom = a[idx + t];

		t = (nz == 0) ? 0 : -SLICE;
		front = a[idx + t];

		t = (nz == (DIMZ-1)) ? 0 : SLICE;
		back = a[idx + t];
		
		c[idx] = C0*center + C1*(left + right + top + bottom + front + back);
	}
}

__global__ void heatflow_shared(float *a, float *c)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	int offset = DIMX*y + x;

	int idx, t, nz;
	float center, left, right, top, bottom, front, back;
	int xl, xr, yt, yb, zf, zb; //left, right, top, bottom, front, back
	// __shared__ float sharedMem[BLKZ+2][BLKY+2][BLKX]; // 9 strides
	__shared__ float sharedMem[BLKZ][BLKY+2][BLKX]; // 3 strides
	
	float *tmp;
	#pragma unroll 
	for(int i=0; i<ILP; i++)
	{
		nz  = ILP*z+i;
		idx = SLICE*(nz) + offset;
		
		xl = (x == 0) 			? 0 : -1;
		xr = (x == (DIMX-1)) 	? 0 : 1;
		yt = (y == 0) 			? 0 : -DIMX;
		yb = (y == (DIMY-1)) 	? 0 : DIMX;
		zf = (nz == 0) 			? 0 : -SLICE;
		zb = (nz == (DIMZ-1)) 	? 0 : SLICE;
		
		
		// ///!!! 86 GFLOPS
		// center 	= a[idx];
		// left 	= a[idx + xl];
		// right 	= a[idx + xr];
		// top 	= a[idx + yt];
		// bottom 	= a[idx + yb];
		// front 	= a[idx + zf];
		// back 	= a[idx + zb];
		
		
		// ///!!! 56 GFLOPS
		// // // center = a[idx];
		// sharedMem[threadIdx.z+1][threadIdx.y+1][threadIdx.x] = a[idx];
		
		// // t = (x == 0) ? 0 : -1;
		// // // left = a[idx + t];
		// // sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x+1-1] = a[idx + t];
		
		// // t = (x == (DIMX-1)) ? 0 : 1;
		// // // right = a[idx + t];
		// // sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x+1+1] = a[idx + t];

		// yt = (y == 0) ? 0 : -DIMX;
		// // // top = a[idx + t];
		// sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x] = a[idx + yt];

		// yb = (y == (DIMY-1)) ? 0 : DIMX;
		// // // bottom = a[idx + t];
		// sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x] = a[idx + yb];

		// zf = (nz == 0) ? 0 : -SLICE;
		// // // front = a[idx + t];
		// sharedMem[threadIdx.z+1-1][threadIdx.y+1+0][threadIdx.x] = a[idx + zf];

		// zb = (nz == (DIMZ-1)) ? 0 : SLICE;
		// // // back = a[idx + t];
		// sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x] = a[idx + zb];
		
		
		
		// ///!!! 71 GFLOPS
		// if(i==0)
		// {
			// // // center = a[idx];
			// sharedMem[threadIdx.z+1][threadIdx.y+1][threadIdx.x] = a[idx];
			
			// // t = (x == 0) ? 0 : -1;
			// // // left = a[idx + t];
			// // sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x+1-1] = a[idx + t];
			
			// // t = (x == (DIMX-1)) ? 0 : 1;
			// // // right = a[idx + t];
			// // sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x+1+1] = a[idx + t];

			// yt = (y == 0) ? 0 : -DIMX;
			// // // top = a[idx + t];
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x] = a[idx + yt];

			// yb = (y == (DIMY-1)) ? 0 : DIMX;
			// // // bottom = a[idx + t];
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x] = a[idx + yb];

			// zf = (nz == 0) ? 0 : -SLICE;
			// // // front = a[idx + t];
			// sharedMem[threadIdx.z+1-1][threadIdx.y+1+0][threadIdx.x] = a[idx + zf];

			// zb = (nz == (DIMZ-1)) ? 0 : SLICE;
			// // // back = a[idx + t];
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x] = a[idx + zb];
		// }
		// else
		// {
			// zf = (nz == 0) ? 0 : -SLICE;
			// // // front = a[idx + t];
			// sharedMem[threadIdx.z+1-1][threadIdx.y+1+0][threadIdx.x]// = a[idx + zf];
			// = sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x];
			
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x+0]
			// = sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x];
			
			// yt = (y == 0) ? 0 : -DIMX;
			// // // top = a[idx + t];
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x] = a[idx + yt];

			// yb = (y == (DIMY-1)) ? 0 : DIMX;
			// // // bottom = a[idx + t];
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x] = a[idx + yb];
			
			// zb = (nz == (DIMZ-1)) ? 0 : SLICE;
			// // // back = a[idx + t];
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x] = a[idx + zb];
		// }
		// __syncthreads();


		// ///!!! 71 GFLOPS
		// if(i==0)
		// {
			// // // center = a[idx];
			// sharedMem[threadIdx.z+1][threadIdx.y+1][threadIdx.x] = a[idx];	
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x] = a[idx + yt];
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x] = a[idx + yb];
			// sharedMem[threadIdx.z+1-1][threadIdx.y+1+0][threadIdx.x] = a[idx + zf];
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x] = a[idx + zb];
		// }
		// else
		// {
			// sharedMem[threadIdx.z+1-1][threadIdx.y+1+0][threadIdx.x]// = a[idx + zf];
			// = sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x];
			
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x]
			// = sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x];
			
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x] = a[idx + yt];
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x] = a[idx + yb];
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x] = a[idx + zb];
		// }
		// __syncthreads();
		

		///!!! 69  GFLOPS
		// if(i==0)
		// {
			// sharedMem[threadIdx.z+1-1][threadIdx.y+1-1][threadIdx.x] = a[idx + yt + zf];
			// sharedMem[threadIdx.z+1-1][threadIdx.y+1+0][threadIdx.x] = a[idx +  0 + zf];
			// sharedMem[threadIdx.z+1-1][threadIdx.y+1+1][threadIdx.x] = a[idx + yb + zf];
			
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x] = a[idx + yt + 0];
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x] = a[idx +  0 + 0];
			// sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x] = a[idx + yb + 0];
			
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1-1][threadIdx.x] = a[idx + yt + zb];			
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x] = a[idx +  0 + zb];
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1+1][threadIdx.x] = a[idx + yb + zb];
		// }
		// else
		// {
			// // sharedMem[threadIdx.z+1-1][threadIdx.y+1-1][threadIdx.x] = sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x];
			// // sharedMem[threadIdx.z+1-1][threadIdx.y+1+0][threadIdx.x] = sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x];
			// // sharedMem[threadIdx.z+1-1][threadIdx.y+1+1][threadIdx.x] = sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x];
			
			// // sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x] = sharedMem[threadIdx.z+1+1][threadIdx.y+1-1][threadIdx.x];
			// // sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x] = sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x];
			// // sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x] = sharedMem[threadIdx.z+1+1][threadIdx.y+1+1][threadIdx.x];

			// sharedMem[threadIdx.z+1+1][threadIdx.y+1-1][threadIdx.x] = a[idx + yt + zb];			
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x] = a[idx +  0 + zb];
			// sharedMem[threadIdx.z+1+1][threadIdx.y+1+1][threadIdx.x] = a[idx + yb + zb];
		// }
		// __syncthreads();
		
		// center = sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x];
		// left   = sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x+xl];
		// right  = sharedMem[threadIdx.z+1+0][threadIdx.y+1+0][threadIdx.x+xr];
		// top    = sharedMem[threadIdx.z+1+0][threadIdx.y+1-1][threadIdx.x+0];
		// bottom = sharedMem[threadIdx.z+1+0][threadIdx.y+1+1][threadIdx.x+0];
		// front  = sharedMem[threadIdx.z+1-1][threadIdx.y+1+0][threadIdx.x+0];
		// back   = sharedMem[threadIdx.z+1+1][threadIdx.y+1+0][threadIdx.x+0];
		
		///!!! 86  GFLOPS
		front 	= (i==0) ? a[idx + zf]:center;
		top 	= a[idx + yt];
		left 	= a[idx + xl];
		center 	= (i==0) ? a[idx]	  :back;
		right 	= a[idx + xr];
		bottom 	= a[idx + yb];
		back 	= a[idx + zb];
		
		// ///!!! 71 GFLOPS, not correct
		// sharedMem[0][0][threadIdx.x] =  a[idx + zb + yt];
		// sharedMem[0][1][threadIdx.x] =  a[idx + zb];
		// sharedMem[0][2][threadIdx.x] =  a[idx + zb + yb];
		// __syncthreads();
			
		// front 	= (i==0) ? a[idx + zf]:center;
		// top 	= (i==0) ? a[idx + yt]:sharedMem[0][0][threadIdx.x];
		// left 	= (i==0) ? a[idx + xl]:sharedMem[0][1][threadIdx.x+xl];
		// center 	= (i==0) ? a[idx]	  :back;
		// right 	= (i==0) ? a[idx + xr]:sharedMem[0][1][threadIdx.x+xr];
		// bottom 	= (i==0) ? a[idx + yb]:sharedMem[0][2][threadIdx.x];
		// // back 	= (i==0) ? a[idx + zb]:sharedMem[0][1][threadIdx.x];
		// back 	= sharedMem[0][1][threadIdx.x];
		
		
		
		
		
		c[idx] = C0*center + C1*(left + right + top + bottom + front + back);	
	}
}



int main()
{
	srand ( time(NULL) );

	//int gpuid = rand() % 8;
	//printf("Assigned GPU ID: %d\n", gpuid);
	//cudaSetDevice( gpuid ); 

	// Allocate GPU memory	
	float *d_a, *d_b, *d_c, *h_c;


	h_c = (float*)malloc(sizeof(float)*DIMX*DIMY*DIMZ);
	cudaMalloc((void**)&(d_a),sizeof(float)*DIMX*DIMY*DIMZ);
	cudaMalloc((void**)&(d_b),sizeof(float)*DIMX*DIMY*DIMZ);
	cudaMalloc((void**)&(d_c),sizeof(float)*DIMX*DIMY*DIMZ);
	
	for(int i=0; i<DIMX*DIMY*DIMZ; i++) h_c[i] = (float)( (int)rand() % 10); // 7;

	cudaMemcpy(d_a, h_c, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_b, h_c, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyHostToDevice);
	

	// copy host to device
	//cudaMemcpy(d_tree.info, h_tree.info, sizeof(stat), cudaMemcpyHostToDevice);

	cudaEvent_t begin, end;

	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaEventRecord(begin, 0);


	// call your kernel here
	// dim3 dimB = dim3(512,1,8/ILP); // block size
	dim3 dimB = dim3(BLKX,BLKY,BLKZ); // block size
	dim3 dimG = dim3(DIMX/dimB.x,DIMY/dimB.y,DIMZ/(dimB.z*ILP));

	// parameters for performance eval
	double flops, gbps, nops, nbp;
	int iter = 20;

#define HEATFLOW //VECMUL//

#ifdef VECMUL
	nbp = 4*3; // # of bytes transferred per point
	nops = 1.; // # of flops per point
	
	for(int i=0; i<iter; i++)
	{
		vectorMul<<<dimG,dimB>>>(d_a, d_b, d_c);
	}
#endif

#ifdef HEATFLOW
	nbp = 8*4; // # of bytes transferred per point
	nops = 8.; // # of flops per point
	
	// initialize global memory
	//init<<<dimG,dimB>>>();

	for(int i=0; i<iter; i++)
	{
		// heatflow<<<dimG,dimB>>>(d_a, d_c);
		heatflow_shared<<<dimG,dimB>>>(d_a, d_c);
	}


#endif


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
	float* h_r = new float[DIMX*DIMY*DIMZ];
	float tmp, result;
	for(int z=0; z<DIMZ; z++)
	{
		for(int y=0; y<DIMY; y++)
		{
			for(int x=0; x<DIMX; x++)
			{
				tmp = C1 *    ( h_c[at(x + 1, y + 0, z + 0, DIMX, DIMY, DIMZ)] +
								h_c[at(x - 1, y + 0, z + 0, DIMX, DIMY, DIMZ)] +
								h_c[at(x + 0, y + 1, z + 0, DIMX, DIMY, DIMZ)] +
								h_c[at(x + 0, y - 1, z + 0, DIMX, DIMY, DIMZ)] +
								h_c[at(x + 0, y + 0, z + 1, DIMX, DIMY, DIMZ)] +
								h_c[at(x + 0, y + 0, z - 1, DIMX, DIMY, DIMZ)] );
				result = C0*h_c[at(x + 0, y + 0, z + 0, DIMX, DIMY, DIMZ)]  + tmp;			
				h_r[at(x, y, z, DIMX, DIMY, DIMZ)] 	= result;
			}
		}
	}
	cudaMemcpy(h_c, d_c, sizeof(float)*DIMX*DIMY*DIMZ, cudaMemcpyDeviceToHost);

	for(int x=0; x<DIMX; x++)
	{
		for(int y=0; y<DIMY; y++)
		{
			for(int z=0; z<DIMZ; z++)
			{		
				if(h_r[z*DIMY*DIMX+y*DIMX+x] != h_c[z*DIMY*DIMX+y*DIMX+x])
				{
					printf("Solution does not match at x: %d, y: %d, z: %d\n", x, y, z);
					printf("h_r (%04.4f), h_c (%04.4f)\n", 
						h_r[z*DIMY*DIMX+y*DIMX+x], 
						h_c[z*DIMY*DIMX+y*DIMX+x]);
					// return -1;
					goto cleanup;
				}
			}
		}
	}
	printf("Solution is correct.\n");
	
	// printf("Value %f\n", h_c[0]);
cleanup:	
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	

	return 0;
}
