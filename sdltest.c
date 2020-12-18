#include <stdio.h>
#include <SDL2/SDL.h>
#include <complex.h>
#include <sys/param.h>

#define WINDOW_WIDTH (2000)
#define WINDOW_HEIGHT (2000)

int main (void) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window*   win  = SDL_CreateWindow("sdltest", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Surface*  surf = SDL_CreateRGBSurface(0, WINDOW_WIDTH, WINDOW_HEIGHT, 32, 0, 0, 0, 0);
    SDL_Texture*  tex  = SDL_CreateTextureFromSurface(rend, surf);

    SDL_FreeSurface(surf);
    SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
    SDL_RenderClear(rend);
    SDL_SetRenderDrawColor(rend, 255, 255, 255, 255);
    SDL_RenderCopy(rend, tex, NULL, NULL);

    double complex z, zi;
    double xv, yv;
    int iter = 50;

        for ( int y = 0; y < WINDOW_HEIGHT; y++ ){
    for ( int x = 0; x < WINDOW_WIDTH; x++ ){

            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)
            xv = - 2 + (double) x / WINDOW_WIDTH  * 4;
            yv =   2 - (double) y / WINDOW_HEIGHT * 4;

            zi = xv + yv * I;  // initial Z value
            z = 0;
            int color;
            
            color = 255; // white as default for values that converge to 0

            // Mandelbrot calc for current (x,y) pixel
            for ( int i = 0; i < iter; i++ ) {
                z = z*z + zi;
                if (cabs(z) > 2) {
                    color = MIN(255,(int) 255*i/40);
                    
                    break;
                }
            }
            SDL_SetRenderDrawColor(rend, color, color, color, 255);
            SDL_RenderDrawPoint(rend, x, y);
        }
    }

    SDL_RenderPresent(rend); // final screen render
    SDL_Delay(5000);
    SDL_DestroyWindow(win);
    SDL_Quit();

}

