#ifndef MANDELMAIN_H
#define MANDELMAIN_H

#define INITIAL_WINDOW_WIDTH (2048)
#define INITIAL_WINDOW_HEIGHT (2048)

#ifdef __x86_64__
#include <immintrin.h>
#endif
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include <stdint.h>
#include <sys/param.h>

#include <gmp.h>

enum rendertargets {
    TARGET_CPU,
    TARGET_AVX,
    TARGET_NEON,
    TARGET_CUDASP,
    TARGET_CUDA,
    TARGET_GMP
};

struct RenderSettings {
    uint32_t *outputBuffer;
    int width;
    int height;
    double zoom;
    double xoffset;
    double yoffset;
    unsigned int iterations;
    uint32_t *deviceBuffer;
    int multithreaded;
};

void mandelbrotCPU(struct RenderSettings rs);
#ifdef __AVX__
void mandelbrotAVX(struct RenderSettings rs);
#endif
#ifdef __aarch64__
void mandelbrotNEON(struct RenderSettings rs);
#endif
void mandelbrotGMP(struct RenderSettings rs);
#ifdef USE_CUDA
void mandelbrotCUDA(struct RenderSettings rs);
void mandelbrotCUDAsp(struct RenderSettings rs);
void initCUDA(struct RenderSettings rs);
void freeCUDA();
#endif

extern const int renderers[];
extern const int num_renderers;

// renders into rs.outputBuffer with the given renderer, returns elapsed seconds
double renderMandelbrot(struct RenderSettings rs, int target);

#ifdef USE_GUI
void runGUI(struct RenderSettings rs);
#endif

#endif
