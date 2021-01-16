
#ifndef MANDELMAIN_H
#define MANDELMAIN_H

#define INITIAL_WINDOW_WIDTH (2048)
#define INITIAL_WINDOW_HEIGHT (2048)

#define MAX_SHADER_SIZE 100000

#include "mandelcpu.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_stdinc.h>
#include <immintrin.h>
#include <stdio.h>
#include <sys/param.h>

#include <GL/glew.h>



enum rendertargets {
    TARGET_CPU,
    TARGET_AVX,
    TARGET_CUDA
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
};

void mandelbrotCPU(struct RenderSettings rs);
void mandelbrotAVX(struct RenderSettings rs);
void mandelbrotCUDA(struct RenderSettings rs);
void initCUDA(struct RenderSettings rs);
void freeCUDA();
void renderWindow(SDL_Renderer *rend, SDL_Texture *tex, struct RenderSettings rs);

#endif