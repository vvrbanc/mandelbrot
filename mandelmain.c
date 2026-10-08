#include "mandelmain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
const int renderers[] = {
    TARGET_CPU,
#ifdef __AVX__
    TARGET_AVX,
#endif
#ifdef __aarch64__
    TARGET_NEON,
#endif
#ifdef USE_CUDA
    TARGET_CUDASP,
    TARGET_CUDA,
#endif
    TARGET_GMP,
};
const int num_renderers = (int)(sizeof renderers / sizeof renderers[0]);

double renderMandelbrot(struct RenderSettings rs, int target) {
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    switch (target) {
#ifdef USE_CUDA
    case TARGET_CUDA:
        printf("Renderer: CUDA double precision\n");
        mandelbrotCUDA(rs);
        break;
    case TARGET_CUDASP:
        printf("Renderer: CUDA single precision\n");
        mandelbrotCUDAsp(rs);
        break;
#endif
#ifdef __AVX__
    case TARGET_AVX:
        printf("Renderer: AVX\n");
        mandelbrotAVX(rs);
        break;
#endif
#ifdef __aarch64__
    case TARGET_NEON:
        printf("Renderer: NEON\n");
        mandelbrotNEON(rs);
        break;
#endif
    case TARGET_GMP:
        printf("Renderer: GMP\n");
        mandelbrotGMP(rs);
        break;
    case TARGET_CPU:
        printf("Renderer: CPU\n");
        mandelbrotCPU(rs);
        break;
    default:
        printf("Renderer undefined!\n");
        break;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double duration_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Time: %f \n", duration_sec);

    return duration_sec;
}

void runBenchmark(struct RenderSettings rs) {
    // arbitrarily chosen zoom point for benchmarking
    rs.xoffset = -1.483183768341172;
    rs.zoom = 942335637702.334351;
    rs.iterations = 40000;

    rs.outputBuffer = malloc((size_t)rs.width * rs.height * sizeof(uint32_t));
    if (rs.outputBuffer == NULL) {
        fprintf(stderr, "Couldn't allocate %dx%d output buffer\n", rs.width, rs.height);
        exit(1);
    }

    // benchmark all renderers except GMP
    for (int i = 0; i < num_renderers; i++) {
        if (renderers[i] != TARGET_GMP)
            renderMandelbrot(rs, renderers[i]);
    }

    free(rs.outputBuffer);
}

int main(int argc, char *argv[]) {
    struct RenderSettings rs = {
        .zoom = 1.0,
        .xoffset = 0,
        .yoffset = 0,
        .iterations = 50,
        .multithreaded = 1,
        .width = INITIAL_WINDOW_WIDTH,
        .height = INITIAL_WINDOW_HEIGHT,
    };
    int bench = argc > 1 && strcmp(argv[1], "bench") == 0;

#ifndef USE_GUI
    if (!bench) {
        fprintf(stderr, "Built without GUI support, only \"%s bench\" is available\n", argv[0]);
        return 1;
    }
#endif

#ifdef USE_CUDA
    initCUDA(rs);
#endif

    if (bench)
        runBenchmark(rs);
#ifdef USE_GUI
    else
        runGUI(rs);
#endif

#ifdef USE_CUDA
    freeCUDA();
#endif
    return 0;
}
