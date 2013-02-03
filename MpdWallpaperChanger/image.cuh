#ifndef _IMAGE_CUH
#define _IMAGE_CUH

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <Imlib2.h> 

__global__ void mixImages(DATA32* im1data, DATA32* im2data, float k, DATA32* dstData);
__global__ void padImage(DATA32* src, int srcW, int srcH, DATA32* dst, int dstW, int dstH);
__global__ void bicubicInterpolation(DATA32* src, int srcW, int srcH, DATA32* dst, int dstW, int dstH);

#endif
