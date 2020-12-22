#include "mandelmain.h"
#include <complex.h>

int *deviceBuffer;

void renderWindow(SDL_Renderer *rend, SDL_Texture *tex, struct RenderSettings rs) {
    void *screenbuf;
    int pitch;
    SDL_LockTexture(tex, NULL, &screenbuf, &pitch);

    rs.outputBuffer = screenbuf;

    // mandelbrotCUDA(rs);

    mandelbrotCPU(rs);

    SDL_UnlockTexture(tex);

    SDL_RenderClear(rend);
    SDL_RenderCopy(rend, tex, NULL, NULL);

    SDL_RenderPresent(rend);

}

int main(void)
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *win =
        SDL_CreateWindow("sdltest", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                         SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *tex = SDL_CreateTexture(rend, SDL_PIXELFORMAT_ARGB8888,
                                         SDL_TEXTUREACCESS_STREAMING,
                                         WINDOW_WIDTH, WINDOW_HEIGHT);

    struct RenderSettings rs;

    rs.x0 = 0;
    rs.y0 = 0;
    rs.zoom = 1.0;
    rs.xoffset = 0;
    rs.yoffset = 0;
    rs.iterations = 20;

    renderWindow(rend, tex, rs);

    int close_requested = 0;
    while (!close_requested)
    {
        SDL_Event event;
        SDL_PollEvent(&event);
        {
            switch (event.type)
            {
            case SDL_QUIT:
                close_requested = 1;
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.scancode)
                {
                case SDL_SCANCODE_ESCAPE:
                    close_requested = 1;
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_KP_MINUS:
                    rs.zoom = rs.zoom / 1.5;
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_KP_DIVIDE:
                    rs.zoom = 1.0;
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_KP_PLUS:
                    rs.zoom = rs.zoom * 1.5;
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_RIGHT:
                    rs.xoffset += (0.2 / rs.zoom);
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_LEFT:
                    rs.xoffset -= (0.2 / rs.zoom);
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_UP:
                    rs.yoffset += (0.2 / rs.zoom);
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_DOWN:
                    rs.yoffset -= (0.2 / rs.zoom);
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_W:
                    rs.iterations = rs.iterations * 2;
                    renderWindow(rend, tex, rs);
                    break;
                case SDL_SCANCODE_S:
                    rs.iterations = rs.iterations / 2;
                    renderWindow(rend, tex, rs);
                    break;
                default:
                    break;
                }
                printf("Zoom: %f\n", rs.zoom);
                printf("Iter: %d\n", rs.iterations);
                break;
            case 1024:
                break;
            default:
                // printf("%d\n", event.type);
                SDL_Delay(16);
            }
        }
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
}
