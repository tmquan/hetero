#ifndef _stencil_3d07_hpp
#define _stencil_3d07_hpp
#include <cuda.h>
void stencil_3d07(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int blockx, int blocky, int blockz, int halo = 0, cudaStream_t stream = 0);

#endif
