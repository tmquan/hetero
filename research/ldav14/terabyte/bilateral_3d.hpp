#ifndef _bilateral_3d_hpp
#define _bilateral_3d_hpp
#include <cuda.h>
void bilateral_3d(unsigned char* deviceSrc, unsigned char* deviceDst, int dimx, int dimy, int dimz, float imageDensity, float colorDensity, int radius, int halo = 0, cudaStream_t stream = 0);

#endif
