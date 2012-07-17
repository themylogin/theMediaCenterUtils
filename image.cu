#include "image.cuh"

__global__ void mixImages(DATA32* im1data, DATA32* im2data, float k, DATA32* dstData)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    unsigned char* currentPixel1 = (unsigned char *)(im1data + n);
    unsigned char* currentPixel2 = (unsigned char *)(im2data + n);
    unsigned char* dstPixel      = (unsigned char *)(dstData + n);
    
    dstPixel[0] = currentPixel1[0] * k + currentPixel2[0] * (1.0 - k);
    dstPixel[1] = currentPixel1[1] * k + currentPixel2[1] * (1.0 - k);
    dstPixel[2] = currentPixel1[2] * k + currentPixel2[2] * (1.0 - k);
}

__global__ void padImage(DATA32* src, int srcW, int srcH, DATA32* dst, int dstW, int dstH)
{
    int xStart = (dstW - srcW) / 2;
    int xEnd = xStart + srcW;

    int yStart = (dstH - srcH) / 2;
    int yEnd = yStart + srcH;
    
    for (int x = 0; x < dstW; x++)
    {
        for (int y = 0; y < dstH; y++)
        {
            if (x < xStart || x > xEnd || y < yStart || y > yEnd)
            {
                dst[y * dstW + x] = 0;
            }
            else
            {
                dst[y * dstW + x] = src[(y - yStart) * srcW + (x - xStart)];
            }
        }
    }
}

__global__ void bicubicInterpolation(DATA32* src, int srcW, int srcH, DATA32* dst, int dstW, int dstH)
{
    int srcX = blockIdx.x * blockDim.x + threadIdx.x;
    int srcY = blockIdx.y * blockDim.y + threadIdx.y;
    if (srcX >= srcW || srcY >= srcH)
    {
        return;
    }
    
    unsigned char p[4][4][3];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                int x = srcX + i;
                if (x >= srcW)
                {
                    x = srcW - 1;
                }
                
                int y = srcY + j;
                if (y >= srcH)
                {
                    y = srcH - 1;
                }
                
                unsigned char* srcPixel = (unsigned char*)(src + y * srcW + x);
                p[i][j][k] = srcPixel[k];
            }
        }
    }
    
    float a[4][4][3];
    for (int i = 0; i < 3; i++)
        a[0][0][i]  = p[1][1][i];
    for (int i = 0; i < 3; i++)
        a[0][1][i]  = -.5 * p[1][0][i] + .5 * p[1][2][i];
    for (int i = 0; i < 3; i++)
        a[0][2][i]  = p[1][0][i] - 2.5 * p[1][1][i] + 2 * p[1][2][i] - .5 * p[1][3][i];
    for (int i = 0; i < 3; i++)
        a[0][3][i]  = -.5 * p[1][0][i] + 1.5 * p[1][1][i] - 1.5 * p[1][2][i] + .5 * p[1][3][i];
    for (int i = 0; i < 3; i++)
        a[1][0][i]  = -.5 * p[0][1][i] + .5 * p[2][1][i];
    for (int i = 0; i < 3; i++)
        a[1][1][i]  = .25 * p[0][0][i] - .25 * p[0][2][i] - .25 * p[2][0][i] + .25 * p[2][2][i];
    for (int i = 0; i < 3; i++)
        a[1][2][i]  = -.5 * p[0][0][i] + 1.25 * p[0][1][i] - p[0][2][i] + .25 * p[0][3][i]
        +  .5 * p[2][0][i] - 1.25 * p[2][1][i] + p[2][2][i] - .25 * p[2][3][i];
    for (int i = 0; i < 3; i++)
        a[1][3][i]  = .25 * p[0][0][i] - .75 * p[0][1][i] + .75 * p[0][2][i] - .25 * p[0][3][i]
        - .25 * p[2][0][i] + .75 * p[2][1][i] - .75 * p[2][2][i] + .25 * p[2][3][i];
    for (int i = 0; i < 3; i++)
        a[2][0][i]  = p[0][1][i] - 2.5 * p[1][1][i] + 2 * p[2][1][i] - .5 * p[3][1][i];
    for (int i = 0; i < 3; i++)
        a[2][1][i]  = -.5 * p[0][0][i] + .5 * p[0][2][i] + 1.25 * p[1][0][i] - 1.25 * p[1][2][i]
        - p[2][0][i] + p[2][2][i] + .25 * p[3][0][i] - .25 * p[3][2][i];
    for (int i = 0; i < 3; i++)
        a[2][2][i]  = p[0][0][i] - 2.5 * p[0][1][i] + 2 * p[0][2][i] - .5 * p[0][3][i] - 2.5 * p[1][0][i]
        + 6.25 * p[1][1][i] - 5 * p[1][2][i] + 1.25 * p[1][3][i] + 2 * p[2][0][i]
        - 5 * p[2][1][i] + 4 * p[2][2][i] - p[2][3][i] - .5 * p[3][0][i]
        + 1.25 * p[3][1][i] - p[3][2][i] + .25 * p[3][3][i];
    for (int i = 0; i < 3; i++)
        a[2][3][i]  = -.5 * p[0][0][i] + 1.5 * p[0][1][i] - 1.5 * p[0][2][i] + .5 * p[0][3][i]
        + 1.25 * p[1][0][i] - 3.75 * p[1][1][i] + 3.75 * p[1][2][i]
        - 1.25 * p[1][3][i] - p[2][0][i] + 3 * p[2][1][i] - 3 * p[2][2][i] + p[2][3][i]
        + .25 * p[3][0][i] - .75 * p[3][1][i] + .75 * p[3][2][i] - .25 * p[3][3][i];
    for (int i = 0; i < 3; i++)
        a[3][0][i]  = -.5 * p[0][1][i] + 1.5 * p[1][1][i] - 1.5 * p[2][1][i] + .5 * p[3][1][i];
    for (int i = 0; i < 3; i++)
        a[3][1][i]  = .25 * p[0][0][i] - .25 * p[0][2][i] - .75 * p[1][0][i] + .75 * p[1][2][i]
        + .75 * p[2][0][i] - .75 * p[2][2][i] - .25 * p[3][0][i] + .25 * p[3][2][i];
    for (int i = 0; i < 3; i++)
        a[3][2][i]  = -.5 * p[0][0][i] + 1.25 * p[0][1][i] - p[0][2][i] + .25 * p[0][3][i]
        + 1.5 * p[1][0][i] - 3.75 * p[1][1][i] + 3 * p[1][2][i] - .75 * p[1][3][i]
        - 1.5 * p[2][0][i] + 3.75 * p[2][1][i] - 3 * p[2][2][i] + .75 * p[2][3][i]
        + .5 * p[3][0][i] - 1.25 * p[3][1][i] + p[3][2][i] - .25 * p[3][3][i];
    for (int i = 0; i < 3; i++)
        a[3][3][i]  = .25 * p[0][0][i] - .75 * p[0][1][i] + .75 * p[0][2][i] - .25 * p[0][3][i]
        - .75 * p[1][0][i] + 2.25 * p[1][1][i] - 2.25 * p[1][2][i] + .75 * p[1][3][i]
        + .75 * p[2][0][i] - 2.25 * p[2][1][i] + 2.25 * p[2][2][i] - .75 * p[2][3][i]
        - .25 * p[3][0][i] + .75 * p[3][1][i] - .75 * p[3][2][i] + .25 * p[3][3][i];
    
    float hx = (float)dstW / (float)srcW;
    float hy = (float)dstH / (float)srcH;
    for (int x = 0; x < hx; x++)
    {
        for (int y = 0; y < hy; y++)
        {
            float _x = (float)x / hx;
            float _y = (float)y / hy;
            
            float _x2 = _x * _x;
            float _x3 = _x2 * _x;
            float _y2 = _y * _y;
            float _y3 = _y2 * _y;
            
            int dstX = srcX * hx + x;
            int dstY = srcY * hy + y;
            unsigned char* dstPixel = (unsigned char*)(dst + dstY * dstW + dstX);
            for (int i = 0; i < 3; i++)
            {
                int value =  (a[0][0][i] + a[0][1][i] * _y + a[0][2][i] * _y2 + a[0][3][i] * _y3) +
                (a[1][0][i] + a[1][1][i] * _y + a[1][2][i] * _y2 + a[1][3][i] * _y3) * _x +
                (a[2][0][i] + a[2][1][i] * _y + a[2][2][i] * _y2 + a[2][3][i] * _y3) * _x2 +
                (a[3][0][i] + a[3][1][i] * _y + a[3][2][i] * _y2 + a[3][3][i] * _y3) * _x3;
                if (value < 0)
                {
                    value = 0;
                }
                if (value > 255)
                {
                    value = 255;
                }
                dstPixel[i] = value;
            }
        }
    }
}
