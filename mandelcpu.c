
#include "mandelmain.h"



int renderMandelThread(void *thd)
{

    struct ThreadData * td = (struct ThreadData *) thd;
    struct RenderSettings rs = * td->renderSettings;

    double complex z, zi;
    double xval, yval;
    Uint32 color;
    Uint32 colorbias;
    Uint32 *dst;

    double x1 = rs.x0 - 2.0 / rs.zoom + rs.xoffset;
    double x2 = rs.x0 + 2.0 / rs.zoom + rs.xoffset;
    double y1 = rs.y0 + 2.0 / rs.zoom + rs.yoffset;
    double y2 = rs.y0 - 2.0 / rs.zoom + rs.yoffset;

    double xpitch = (x2 - x1) / WINDOW_WIDTH;
    double ypitch = (y1 - y2) / WINDOW_HEIGHT;

    dst = (Uint32 *)((Uint8 *)rs.outputBuffer + td->startRow * WINDOW_WIDTH * 4);

    for (int y = td->startRow; y <= td->endRow; y++)
    {
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {

            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)

            xval = x1 + xpitch * x;
            yval = y1 - ypitch * y;

            zi = xval + yval * I; // initial Z value
            z = 0;

            color = 0xffffffff; // white as default for values that converge to 0

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 1; i <= rs.iterations; i++)
            {
                z = z * z + zi;
                if (cabs(z) > 2)
                {
                    colorbias = MIN(255, (int)255 * i / rs.iterations * 2);
                    color = (0xFF000000 | (colorbias << 16) | (colorbias << 8) | colorbias);
                    break;
                }
            }
            *dst++ = color;
        }
    }
    return 0;
}


void mandelbrotCPU(struct RenderSettings rs) {

    int threadNum = 32;

    SDL_Thread * threadarray[threadNum];
    struct ThreadData td[threadNum];

    for (int i = 0; i < threadNum; i++ ) {
        td[i].renderSettings = &rs;
        td[i].startRow = i*WINDOW_HEIGHT/threadNum;
        td[i].endRow = (1+i)*WINDOW_HEIGHT/threadNum -1;
        threadarray[i] = SDL_CreateThread(renderMandelThread, "mandelThread", (void *) &td[i]);
    }

    for (int i = 0; i < threadNum; i++ ) {
        SDL_WaitThread(threadarray[i], NULL);
    }

};
