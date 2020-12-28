#include "mandelmain.h"
#include <complex.h>
#include <time.h>

// int rendertarget = TARGET_AVX;
int rendertarget = TARGET_CUDA;
// int rendertarget = TARGET_CPU;

void renderWindow(SDL_Renderer *rend, SDL_Texture *tex, struct RenderSettings rs) {
    void *screenbuf;
    int pitch;
    SDL_LockTexture(tex, NULL, &screenbuf, &pitch);

    rs.outputBuffer = screenbuf;

    time_t start, end;
    struct timespec curTime;

    clock_gettime(CLOCK_REALTIME, &curTime);
    start = curTime.tv_sec * 1000000000 + curTime.tv_nsec;

    switch (rendertarget) {
    case TARGET_CUDA:
        mandelbrotCUDA(rs);
        break;
    case TARGET_AVX:
        mandelbrotAVX(rs);
        break;
    case TARGET_CPU:
        mandelbrotCPU(rs);
        break;
    default:
        mandelbrotCPU(rs);
        break;
    }

    clock_gettime(CLOCK_REALTIME, &curTime);
    end = curTime.tv_sec * 1000000000 + curTime.tv_nsec;

    double duration_sec = (double)(end - start) / 1000000000.0;

    printf("Time: %f \n", duration_sec);

    SDL_UnlockTexture(tex);

    SDL_RenderClear(rend);
    SDL_RenderCopy(rend, tex, NULL, NULL);

    SDL_RenderPresent(rend);
}

int main(int argc, char *argv[]) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *win =
        SDL_CreateWindow("sdltest", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                         SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *tex = SDL_CreateTexture(rend, SDL_PIXELFORMAT_ARGB8888,
                                         SDL_TEXTUREACCESS_STREAMING,
                                         WINDOW_WIDTH, WINDOW_HEIGHT);

    struct RenderSettings rs;
    initCUDA();

    rs.zoom = 1.0;
    rs.xoffset = 0;
    rs.yoffset = 0;
    rs.iterations = 50;
    rs.width = WINDOW_WIDTH;
    rs.height = WINDOW_HEIGHT;
    // rs.xoffset = -1.456241426611797;
    // rs.yoffset = -0.070233196159122;
    // rs.zoom = 11.390625;
    // rs.iterations = 50;

    // rs.xoffset = -1.484935782949454;
    // rs.zoom = 16585998.481410;
    // rs.iterations = 5000;
    int close_requested = 0;

    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        close_requested = 1;

        // rs.xoffset = -1.456241426611797;
        // rs.yoffset = -0.070233196159122;
        // rs.zoom = 11.390625;
        // rs.iterations = 50;

        rs.xoffset = -1.484935782949454;
        rs.zoom = 16585998.481410;
        rs.iterations = 5000;

        // rendertarget = TARGET_CPU;
        // renderWindow(rend, tex, rs);
        // rendertarget = TARGET_AVX;
        // renderWindow(rend, tex, rs);
        rendertarget = TARGET_CUDA;
    }

    renderWindow(rend, tex, rs);

    while (!close_requested) {
        SDL_Event event;
        SDL_PollEvent(&event);
        {
            switch (event.type) {
            case SDL_QUIT:
                close_requested = 1;
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.scancode) {
                case SDL_SCANCODE_ESCAPE:
                    close_requested = 1;
                    break;
                case SDL_SCANCODE_KP_MINUS:
                    if (rs.zoom > 0.5)
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
                case SDL_SCANCODE_TAB:
                    if (rendertarget == 2)
                        rendertarget = 0;
                    else
                        rendertarget++;

                    renderWindow(rend, tex, rs);
                    SDL_Delay(100);
                    break;
                default:
                    break;
                }
                printf("Xoffset: %.15f\n", rs.xoffset);
                printf("Yoffset: %.15f\n", rs.yoffset);
                printf("Zoom: %f\n", rs.zoom);
                printf("Iter: %d\n", rs.iterations);
                printf("rendertarget: %d\n", rendertarget);
                printf("\n");
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
