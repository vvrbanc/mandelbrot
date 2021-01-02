
#ifndef MANDELMAIN_H
#define MANDELMAIN_H

#include "mandelcpu.h"
#include "settings.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_stdinc.h>
#include <immintrin.h>
#include <stdio.h>
#include <sys/param.h>

enum rendertargets {
    TARGET_CPU,
    TARGET_AVX,
    TARGET_CUDA
};

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

void mandelbrotCPU(struct RenderSettings rs);
void mandelbrotAVX(struct RenderSettings rs);
void mandelbrotCUDA(struct RenderSettings rs);
void initCUDA();
void freeCUDA();
void renderWindow(SDL_Renderer *rend, SDL_Texture *tex, struct RenderSettings rs);

#endif