#ifndef _stencil_3d_hpp
#define _stencil_3d_hpp
#include <cuda.h>
void stencil_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int halo = 0, cudaStream_t stream = 0);

#endif