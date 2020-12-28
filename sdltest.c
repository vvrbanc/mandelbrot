#include <SDL2/SDL.h>
#include <complex.h>
#include <stdio.h>
#include <sys/param.h>

#define WINDOW_WIDTH (2048)
#define WINDOW_HEIGHT (2048)
#define MAXITER (50)

static int renderMandelThread(void *threadData);

struct ThreadData {
    Uint32 *screenbuf;
    Uint32 startRow;
    Uint32 endRow;
};

int main(void) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *win = SDL_CreateWindow("sdltest", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *tex = SDL_CreateTexture(rend, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, WINDOW_HEIGHT);

    void *screenbuf;
    int pitch;
    SDL_LockTexture(tex, NULL, &screenbuf, &pitch);

    struct ThreadData td1, td2;

    td1.screenbuf = screenbuf;
    td1.startRow = 0;
    td1.endRow = 1023;

    td2.screenbuf = screenbuf;
    td2.startRow = 1024;
    td2.endRow = 2047;

    SDL_Thread *thread1;
    SDL_Thread *thread2;

    thread1 = SDL_CreateThread(renderMandelThread, "mandelThread1", &td1);
    thread2 = SDL_CreateThread(renderMandelThread, "mandelThread2", &td2);

    SDL_WaitThread(thread1, NULL);
    SDL_WaitThread(thread2, NULL);

    SDL_UnlockTexture(tex);

    SDL_RenderClear(rend);
    SDL_RenderCopy(rend, tex, NULL, NULL);

    SDL_RenderPresent(rend); // final screen render
    SDL_Delay(5000);
    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
}

static int renderMandelThread(void *threadData) {

    struct ThreadData *td = threadData;

    double complex z, zi;
    double xv, yv;
    Uint32 color;
    Uint32 colorbias;
    Uint32 *dst;

    dst = (Uint32 *)((Uint8 *)td->screenbuf + td->startRow * WINDOW_WIDTH * 4);

    for (int y = td->startRow; y <= td->endRow; y++) {
        for (int x = 0; x < WINDOW_WIDTH; x++) {

            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)
            xv = -2 + (double)x / WINDOW_WIDTH * 4;
            yv = 2 - (double)y / WINDOW_HEIGHT * 4;

            zi = xv + yv * I; // initial Z value
            z = 0;

            color = 0xff000000; // black as default for values that converge to 0

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 0; i < MAXITER; i++) {
                z = z * z + zi;
                if (cabs(z) > 2) {
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
