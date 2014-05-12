#ifndef _threshold_3d_hpp
#define _threshold_3d_hpp
#include <cuda.h>
void threshold_3d(float* deviceSrc, float* deviceDst, int dimx, int dimy, int dimz, int radius, float thresh, int halo = 0, cudaStream_t stream = 0);

#endif
