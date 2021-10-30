#include <bits/stdint-uintn.h>
#include <stdio.h>
#include <sys/param.h>

struct RenderSettings {
    uint32_t *outputBuffer;
    int width;
    int height;
    double zoom;
    double xoffset;
    double yoffset;
    unsigned int iterations;
    uint32_t *deviceBuffer;
};

uint32_t *deviceBuffer;
char cudaInitialized = 0;

__global__ void mandelbrotCalc(struct RenderSettings rs) {
    int *deviceBuffer = (int *)rs.deviceBuffer;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double cReal, cImag, zReal, zImag, z2Real, z2Imag, zrzi;
    int color;
    int colorbias;

    double x1 = rs.xoffset - 2.0 / rs.zoom * rs.width / rs.height;
    double x2 = rs.xoffset + 2.0 / rs.zoom * rs.width / rs.height;
    double y1 = rs.yoffset + 2.0 / rs.zoom;

    double pixel_pitch = (x2 - x1) / rs.width;

    int x, y;

    for (int w = index; w < rs.height * rs.width; w += stride) {

        y = w / rs.width;
        if (y > 0) {
            x = w % rs.width;
        } else {
            x = w;
        }

        cImag = y1 - pixel_pitch * y;
        cReal = x1 + pixel_pitch * x;

        zReal = cReal;
        zImag = cImag;

        color = 0x000000FF; // black as default for values that converge to 0

        for (int i = 0; i < rs.iterations; i++) {
            z2Real = zReal * zReal;
            z2Imag = zImag * zImag;
            zrzi = zReal * zImag;

            zReal = cReal + z2Real - z2Imag;
            zImag = zrzi + zrzi + cImag;

            if (z2Real + z2Imag > 4.0f) {
                colorbias = MIN(255, i * 510.0 / rs.iterations);
                color = (color | (colorbias << 24) | (colorbias << 16) | colorbias << 8);
                break;
            }
        }
        deviceBuffer[w] = color;
    }
}

__global__ void mandelbrotCalcSP(struct RenderSettings rs) {
    int *deviceBuffer = (int *)rs.deviceBuffer;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float cReal, cImag, zReal, zImag, z2Real, z2Imag, zrzi;
    int color;
    int colorbias;

    float x1 = rs.xoffset - 2.0 / rs.zoom * rs.width / rs.height;
    float x2 = rs.xoffset + 2.0 / rs.zoom * rs.width / rs.height;
    float y1 = rs.yoffset + 2.0 / rs.zoom;

    float pixel_pitch = (x2 - x1) / rs.width;

    int x, y;

    for (int w = index; w < rs.height * rs.width; w += stride) {

        y = w / rs.width;
        if (y > 0) {
            x = w % rs.width;
        } else {
            x = w;
        }

        cImag = y1 - pixel_pitch * y;
        cReal = x1 + pixel_pitch * x;

        zReal = cReal;
        zImag = cImag;

        color = 0x000000FF; // black as default for values that converge to 0

        for (int i = 0; i < rs.iterations; i++) {
            z2Real = zReal * zReal;
            z2Imag = zImag * zImag;
            zrzi = zReal * zImag;

            zReal = cReal + z2Real - z2Imag;
            zImag = zrzi + zrzi + cImag;

            if (z2Real + z2Imag > 4.0f) {
                colorbias = MIN(255, i * 510.0 / rs.iterations);
                color = (color | (colorbias << 24) | (colorbias << 16) | colorbias << 8);
                break;
            }
        }
        deviceBuffer[w] = color;
    }
}

extern "C" void freeCUDA() {
    if (cudaInitialized == 1) {
        cudaFree(deviceBuffer);
        cudaInitialized = 0;
    }
}

extern "C" void initCUDA(struct RenderSettings rs) {
    // allocates device buffer on first run
    // destroys and re-allocates buffer if window dimensions change

    static int width = 0;
    static int height = 0;
    if (cudaInitialized == 0) {
        cudaMalloc((void **)&deviceBuffer, rs.width * rs.height * 4);
        width = rs.width;
        height = rs.height;
        cudaInitialized = 1;
    } else {
        if (rs.width != width || rs.height != height) {
            freeCUDA();
            initCUDA(rs);
        }
    }
}

extern "C" void mandelbrotCUDA(struct RenderSettings rs) {
    initCUDA(rs);
    uint32_t *screenBuffer = rs.outputBuffer;
    rs.deviceBuffer = deviceBuffer;
    mandelbrotCalc<<<2048, 1024>>>(rs);
    cudaDeviceSynchronize();
    cudaMemcpy(screenBuffer, deviceBuffer, rs.width * rs.height * 4, cudaMemcpyDeviceToHost);
}

extern "C" void mandelbrotCUDAsp(struct RenderSettings rs) {
    initCUDA(rs);
    uint32_t *screenBuffer = rs.outputBuffer;
    rs.deviceBuffer = deviceBuffer;
    mandelbrotCalcSP<<<2048, 1024>>>(rs);
    cudaDeviceSynchronize();
    cudaMemcpy(screenBuffer, deviceBuffer, rs.width * rs.height * 4, cudaMemcpyDeviceToHost);
}
