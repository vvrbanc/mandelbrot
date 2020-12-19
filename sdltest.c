#include <stdio.h>
#include <SDL2/SDL.h>
#include <complex.h>
#include <sys/param.h>

#define WINDOW_WIDTH (2000)
#define WINDOW_HEIGHT (2000)

void renderMandel(Uint64 *screenbuf);

int main (void) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window*   win  = SDL_CreateWindow("sdltest", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture * tex = SDL_CreateTexture(rend, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, WINDOW_HEIGHT);

    void *screenbuf;
    int pitch;
    SDL_LockTexture(tex, NULL, &screenbuf, &pitch);

    renderMandel(screenbuf);


    // for (row = 0; row < MOOSEPIC_H; ++row) {
    //     dst = (Uint32*)((Uint8*)pixels + row * pitch);
    //     for (col = 0; col < MOOSEPIC_W; ++col) {
    //         color = &MooseColors[*src++];
    //         *dst++ = (0xFF000000|(color->r<<16)|(color->g<<8)|color->b);
    //     }
    // }    
    // http://hg.libsdl.org/SDL/file/default/test/teststreaming.c
    // https://developer.nvidia.com/blog/even-easier-introduction-cuda/
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

    SDL_UnlockTexture(tex);

    SDL_RenderClear(rend);
    SDL_RenderCopy(rend, tex, NULL, NULL);

    SDL_RenderPresent(rend); // final screen render
    SDL_Delay(3000);
    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();

}

void renderMandel(Uint64 *screenbuf) {
    Uint64 *sb;
    sb = (Uint64*)((Uint8*)screenbuf + 1000 * 8000 + 1000 * 4);
    *sb = 0xffffffff;

    double complex z, zi;
    double xv, yv;
    int iter = 50;

    for ( int x = 0; x < WINDOW_WIDTH; x++ ){
        for ( int y = 0; y < WINDOW_HEIGHT; y++ ){

            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)
            xv = - 2 + (double) x / WINDOW_WIDTH  * 4;
            yv =   2 - (double) y / WINDOW_HEIGHT * 4;

            zi = xv + yv * I;  // initial Z value
            z = 0;
            Uint32 color;
            Uint32 colorbias;
            
            color = 0xffffffff; // white as default for values that converge to 0



            // Mandelbrot calc for current (x,y) pixel
            for ( int i = 0; i < iter; i++ ) {
                z = z*z + zi;
                if (cabs(z) > 2) {
                    colorbias = MIN(255,(int) 255*i/40);
                    color = 0xFF000000 + colorbias * 0x10000 + colorbias * 0x100 + colorbias;
                    break;
                }
            }
            sb = (Uint64*)((Uint8*)screenbuf + y * 8000 + x*4);
            *sb = color;
        }
    }
}
