#ifndef _bilateral_3d_hpp
#define _bilateral_3d_hpp
#include <cuda.h>
void bilateral_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int radius, int halo = 0, cudaStream_t stream = 0);

#endif
