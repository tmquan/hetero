#ifndef _stencil_3d_hpp
#define _stencil_3d_hpp
#include <cuda.h>
// __global__ void stencil_3d(float *src, float *dst, int dimx, int dimy, int dimz, int slice, int ilp, int halo);
__global__ void stencil_3d(float *src, float *dst, int dimx, int dimy, int dimz, int slice, int ilp, int halo, float C0, float C1);
// void stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo = 0, cudaStream_t stream = 0);
// void stencil_3d_naive_7points(cudaArray* deviceSrc, cudaArray* deviceDst, int dimx, int dimy, int dimz, int halo = 0, cudaStream_t stream = 0);
// void stencil_3d_naive_7points(cudaPitchedPtr deviceSrc, cudaPitchedPtr deviceDst, int dimx, int dimy, int dimz, int halo = 0, cudaStream_t stream = 0);

// void stencil_3d_global(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo = 0, cudaStream_t stream = 0);


#endif