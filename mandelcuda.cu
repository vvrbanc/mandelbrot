#include "settings.h"
#include <sys/param.h>

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

__global__ void mandelbrotCalc(struct RenderSettings rs) {

  int *deviceBuffer = (int *)rs.deviceBuffer;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double cReal, cImag, zReal, zImag, z2Real, z2Imag;
  int color;
  int colorbias;

  double x1 = rs.x0 - 2.0 / rs.zoom + rs.xoffset;
  double x2 = rs.x0 + 2.0 / rs.zoom + rs.xoffset;
  double y1 = rs.y0 + 2.0 / rs.zoom + rs.yoffset;
  double y2 = rs.y0 - 2.0 / rs.zoom + rs.yoffset;

  double xpitch = (x2 - x1) / WINDOW_WIDTH;
  double ypitch = (y1 - y2) / WINDOW_HEIGHT;

  for (int y = index; y < WINDOW_HEIGHT; y += stride) {

    for (int x = 0; x < WINDOW_WIDTH; x++) {

      cReal = x1 + xpitch * x;
      cImag = y1 - ypitch * y;

      zReal = cReal;
      zImag = cImag;

      color = 0; // white as default for values that converge to 0

      for (int i = 0; i < rs.iterations; i++) {
        z2Real = zReal * zReal - zImag * zImag + cReal;
        z2Imag = 2.0f * zReal * zImag + cImag;

        zReal = z2Real;
        zImag = z2Imag;

        if (zReal * zReal + zImag * zImag > 4.0f) {
          colorbias = MIN(255, (int)255 * i / rs.iterations * 2);
          color =
              (0xFF000000 | (colorbias << 16) | (colorbias << 8) | colorbias);
          break;
        }
      }

      deviceBuffer[y * WINDOW_HEIGHT + x] = color;
    }
  }
}

extern "C" void mandelbrotCUDA(struct RenderSettings rs) {

  unsigned int *screenBuffer = rs.outputBuffer;

  cudaMalloc((void **)&deviceBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * 4);
  rs.deviceBuffer = (long int)deviceBuffer;
  mandelbrotCalc<<<16, 128>>>(rs);
  cudaDeviceSynchronize();
  cudaMemcpy(screenBuffer, deviceBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * 4,
             cudaMemcpyDeviceToHost);
  cudaFree(deviceBuffer);
}
