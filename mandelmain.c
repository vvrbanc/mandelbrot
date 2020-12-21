#include <stdio.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_stdinc.h>
#include <complex.h>
#include <sys/param.h>
#include "settings.h"
#include "mandelmain.h"

int *deviceBuffer;

struct ThreadData
{
    Uint32 *outputBuffer;
    Uint32 startRow;
    Uint32 endRow;
};

int renderMandelThread(void *threadData)
{

    struct ThreadData *td = threadData;

    double complex z, zi;
    double xv, yv;
    Uint32 color;
    Uint32 colorbias;
    Uint32 *dst;

    dst = (Uint32 *)((Uint8 *)td->outputBuffer + td->startRow * WINDOW_WIDTH * 4);

    for (int y = td->startRow; y <= td->endRow; y++)
    {
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {

            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)
            xv = -2 + (double)x / WINDOW_WIDTH * 4;
            yv = 2 - (double)y / WINDOW_HEIGHT * 4;

            zi = xv + yv * I; // initial Z value
            z = 0;

            color = 0xffffffff; // white as default for values that converge to 0

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 0; i < MAXITER; i++)
            {
                z = z * z + zi;
                if (cabs(z) > 2)
                {
                    colorbias = MIN(255, (int)255 * i / 40);
                    color = (0xFF000000 | (colorbias << 16) | (colorbias << 8) | colorbias);
                    break;
                }
            }
            *dst++ = color;
        }
    }
    return 0;
}


void mandelbrotCPU(void *outputBuffer ) {

    int tn = 32;

    SDL_Thread * threadarray[tn];
    struct ThreadData td[tn];

    for (int i = 0; i < tn; i++ ) {
        td[i].outputBuffer = outputBuffer;
        td[i].startRow = i*WINDOW_HEIGHT/tn;
        td[i].endRow = (1+i)*WINDOW_HEIGHT/tn -1;
        threadarray[i] = SDL_CreateThread(renderMandelThread, "mandelThread", &td[i]);
    }

    for (int i = 0; i < tn; i++ ) {
        SDL_WaitThread(threadarray[i], NULL);
    }

};

int renderMandel2(void *threadData)
{

    struct ThreadData *td = threadData;

    double complex z, zi;
    double xv, yv;
    Uint32 color;
    Uint32 colorbias;
    Uint32 *dst;

    dst = (Uint32 *)((Uint8 *)td->outputBuffer + td->startRow * WINDOW_WIDTH * 4);

    for (int y = td->startRow; y <= td->endRow; y++)
    {
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {

            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)
            xv = -2 + (double)x / WINDOW_WIDTH * 4;
            yv = 2 - (double)y / WINDOW_HEIGHT * 4;

            zi = xv + yv * I; // initial Z value
            z = 0;

            color = 0xffffffff; // white as default for values that converge to 0

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 0; i < MAXITER; i++)
            {
                z = z * z + zi;
                if (cabs(z) > 2)
                {
                    colorbias = MIN(255, (int)255 * i / 40);
                    color = (0xFF000000 | (colorbias << 16) | (colorbias << 8) | colorbias);
                    break;
                }
            }
            *dst++ = color;
        }
    }
    return 0;
}

void mandelbrotCPUTest(void *outputBuffer ) {
    

};


int main(void)
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *win = SDL_CreateWindow("sdltest", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE
    );
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *tex = SDL_CreateTexture(rend, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, WINDOW_HEIGHT);

    void *screenbuf;
    int pitch;
    SDL_LockTexture(tex, NULL, &screenbuf, &pitch);

    // mandelbrotCUDA(screenbuf);

    mandelbrotCPU(screenbuf);

    SDL_UnlockTexture(tex);

    SDL_RenderClear(rend);
    SDL_RenderCopy(rend, tex, NULL, NULL);

    SDL_RenderPresent(rend); // final screen render
    // SDL_Delay(5000);

    // int close_requested = 0;
    // while (!close_requested) {
    //     SDL_Event event;
    //     SDL_PollEvent(&event);
    //     {
    //         switch (event.type)
    //         {
    //         case SDL_QUIT:
    //             close_requested = 1;
    //             break;
    //         case SDL_KEYDOWN:
    //             switch (event.key.keysym.scancode)
    //             {
    //             case SDL_SCANCODE_ESCAPE:
    //                 close_requested = 1;
    //                 break;
    //             default:
    //                 break;
    //             }
    //             break;
    //         case 1024:
    //             break;                
    //         default:
    //             // printf("%d\n", event.type);
    //             SDL_Delay(16);
    //         }
    //     }
    // }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
}
