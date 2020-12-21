#include <sys/param.h>
#include "settings.h"

int *deviceBuffer;

__global__ void mandelbrotCalc(int *outputBuffer) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float cReal, cImag;
  int color, colorbias;

  for (int y = index; y < WINDOW_HEIGHT; y += stride) {
    for (int x = 0; x < WINDOW_WIDTH; x++) {
      cReal = -2 + (float)x / WINDOW_WIDTH * 4;
      cImag = 2 - (float)y / WINDOW_HEIGHT * 4;

      float zReal = 0.0f, zImag = 0.0, z2Real, z2Imag;
      color = 0xffffffff; // white as default for values that converge to 0

      int i;

      for (i = 0; i < MAXITER; i++) {
        z2Real = zReal * zReal - zImag * zImag + cReal;
        z2Imag = 2.0f * zReal * zImag + cImag;

        zReal = z2Real;
        zImag = z2Imag;

        if (zReal * zReal + zImag * zImag > 4.0f) {
          colorbias = MIN(255, (int)255 * i / 40);
          color =
              (0xFF000000 | (colorbias << 16) | (colorbias << 8) | colorbias);
          break;
        }
      }

      outputBuffer[y * WINDOW_HEIGHT + x] = color;
    }
  }
}

extern "C" void mandelbrotCUDA(int *screenbuffer) {
  cudaMalloc((void **)&deviceBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * 4);
  mandelbrotCalc<<<16, 128>>>(deviceBuffer);
  cudaDeviceSynchronize();
  cudaMemcpy(screenbuffer, deviceBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * 4, cudaMemcpyDeviceToHost);
  cudaFree(deviceBuffer);
}