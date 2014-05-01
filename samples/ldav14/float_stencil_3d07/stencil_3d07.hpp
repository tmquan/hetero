#ifndef _stencil_3d07_hpp
#define _stencil_3d07_hpp
#include <cuda.h>
void stencil_3d07(cudaArray * deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int blockx, int blocky, int blockz, int ilp, int halo = 0, cudaStream_t stream = 0);

#endif
