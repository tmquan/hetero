#ifndef _minimum_3d_hpp
#define _minimum_3d_hpp
#include <cuda.h>
void minimum_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int radius, int halo = 0, cudaStream_t stream = 0);

#endif
