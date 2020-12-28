#include <sys/param.h>

#include "settings.h"

struct RenderSettings {
    unsigned int *outputBuffer;
    double width;
    double height;
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

    double x1 = rs.xoffset - 2.0 / rs.zoom;
    double x2 = rs.xoffset + 2.0 / rs.zoom;
    double y1 = rs.yoffset + 2.0 / rs.zoom;
    double y2 = rs.yoffset - 2.0 / rs.zoom;

    double xpitch = (x2 - x1) / WINDOW_WIDTH;
    double ypitch = (y1 - y2) / WINDOW_HEIGHT;

    int x, y;

    for (int w = index; w < WINDOW_HEIGHT * WINDOW_WIDTH; w += stride) {

        y = w / WINDOW_WIDTH;
        if (y > 0) {
            x = w % WINDOW_WIDTH;
        } else {
            x = w;
        }

        cImag = y1 - ypitch * y;
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
    mandelbrotCalc<<<1024, 1024>>>(rs);
    cudaDeviceSynchronize();
    cudaMemcpy(screenBuffer, deviceBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * 4, cudaMemcpyDeviceToHost);
}
