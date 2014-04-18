#ifndef _stencil_2d_hpp
#define _stencil_2d_hpp
#include <cuda.h>
void stencil_2d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int halo = 0, cudaStream_t stream = 0);

#endif
