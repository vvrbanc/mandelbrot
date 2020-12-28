#include <sys/param.h>

#include "settings.h"

struct RenderSettings {
    unsigned int *outputBuffer;
    double x0;
    double y0;
    double zoom;
    double xoffset;
    double yoffset;
    unsigned int iterations;
    long int deviceBuffer;
};

int *deviceBuffer;
int cudaInitialized = 0;

__global__ void mandelbrotCalc(struct RenderSettings rs) {
    int *deviceBuffer = (int *)rs.deviceBuffer;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double cReal, cImag, zReal, zImag, z2Real, z2Imag, zrzi;
    int color;
    int colorbias;

    double x1 = rs.x0 - 2.0 / rs.zoom + rs.xoffset;
    double x2 = rs.x0 + 2.0 / rs.zoom + rs.xoffset;
    double y1 = rs.y0 + 2.0 / rs.zoom + rs.yoffset;
    double y2 = rs.y0 - 2.0 / rs.zoom + rs.yoffset;

    double xpitch = (x2 - x1) / WINDOW_WIDTH;
    double ypitch = (y1 - y2) / WINDOW_HEIGHT;

    for (int y = index; y < WINDOW_HEIGHT; y += stride) {
        cImag = y1 - ypitch * y;

        for (int x = 0; x < WINDOW_WIDTH; x++) {
            cReal = x1 + xpitch * x;

            zReal = cReal;
            zImag = cImag;

            color = 0; // black as default for values that converge to 0

            for (int i = 0; i < rs.iterations; i++) {
                z2Real = zReal * zReal;
                z2Imag = zImag * zImag;
                zrzi = zReal * zImag;

                zReal = cReal + z2Real - z2Imag;
                zImag = zrzi + zrzi + cImag;

                if (z2Real + z2Imag > 4.0f) {
                    colorbias = MIN(255, i * 510.0 / rs.iterations);
                    color = (0xFF000000 | (colorbias << 16) | (colorbias << 8) | colorbias);
                    break;
                }
            }
            deviceBuffer[y * WINDOW_HEIGHT + x] = color;
        }
    }
}

extern "C" void initCUDA() {
    if (cudaInitialized == 0) {
        cudaMalloc((void **)&deviceBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * 4);
        cudaInitialized = 1;
    }
}

extern "C" void freeCUDA() {
    if (cudaInitialized == 1) {
        cudaFree(deviceBuffer);
        cudaInitialized = 0;
    }
}

extern "C" void mandelbrotCUDA(struct RenderSettings rs) {
    initCUDA();
    unsigned int *screenBuffer = rs.outputBuffer;
    rs.deviceBuffer = (long int)deviceBuffer;
    mandelbrotCalc<<<64, 32>>>(rs);
    cudaDeviceSynchronize();
    cudaMemcpy(screenBuffer, deviceBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * 4, cudaMemcpyDeviceToHost);
}