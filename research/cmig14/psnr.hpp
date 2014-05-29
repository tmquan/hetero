#ifndef PSNR_HPP
#define PSNR_HPP

#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <fstream>


// float getPSNR(float2 *I1, float2 *I2, int dimx, int dimy, int dimz);
float PSNR(float2 *inA, float2 *inB, float *se, int dimx, int dimy, int dimz);
#endif