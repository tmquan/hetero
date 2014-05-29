#ifndef _median_3d_hpp
#define _median_3d_hpp
#include <cuda.h>
void median_3d(unsigned char* deviceSrc, unsigned char* deviceDst, int dimx, int dimy, int dimz, int radius, int halo = 0, cudaStream_t stream = 0);

#endif
