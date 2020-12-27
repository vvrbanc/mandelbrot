
#ifndef MANDELMAIN_H
#define MANDELMAIN_H

#include "mandelcpu.h"
#include "settings.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_stdinc.h>
#include <stdio.h>
#include <sys/param.h>
#include <immintrin.h>

enum rendertargets {
    TARGET_CPU,
    TARGET_AVX,
    TARGET_CUDA
};

struct RenderData
{
    Uint32 *outputBuffer;
    Uint32 startRow;
    Uint32 endRow;
};

struct RenderSettings
{
    unsigned int *outputBuffer;
    double x0;
    double y0;
    double zoom;
    double xoffset;
    double yoffset;
    unsigned int iterations;
};

struct ThreadData
{
    struct RenderSettings *renderSettings;
    Uint32 startRow;
    Uint32 endRow;
};


void mandelbrotCPU(struct RenderSettings rs);
void mandelbrotAVX(struct RenderSettings rs);
void mandelbrotCUDA(struct RenderSettings rs);
int renderMandelThread(void *td);

void renderWindow(SDL_Renderer *rend, SDL_Texture *tex, struct RenderSettings rs);

#endif