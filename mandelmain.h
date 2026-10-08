
#ifndef MANDELMAIN_H
#define MANDELMAIN_H

#define INITIAL_WINDOW_WIDTH (2048)
#define INITIAL_WINDOW_HEIGHT (2048)

#define MAX_SHADER_SIZE 100000

#include "mandelcpu.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_stdinc.h>
#include <cglm/cglm.h>
#ifdef __x86_64__
#include <immintrin.h>
#endif
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include <stdio.h>
#include <sys/param.h>

#include <GL/glew.h>

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
void renderWindow(SDL_Renderer *rend, SDL_Texture *tex, struct RenderSettings rs);

#endif