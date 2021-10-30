#include "mandelmain.h"
#include <time.h>

int rendertarget = TARGET_AVX;

SDL_Window *win;
SDL_Renderer *rend;
SDL_Texture *tex;
SDL_mutex *mutex;
int mutexstatus;
int close_requested = 0;

struct RenderSettings rs;

void renderWindow(SDL_Renderer *rend, SDL_Texture *tex, struct RenderSettings rs) {
    int pitch;

    void *screenbuf;

    mutexstatus = SDL_TryLockMutex(mutex);

    if (mutexstatus != 0) {
        fprintf(stderr, "Couldn't lock mutex\n");
        return;
    }

    SDL_LockTexture(tex, NULL, &screenbuf, &pitch);

    rs.outputBuffer = screenbuf;

    time_t start, end;
    struct timespec curTime;

    clock_gettime(CLOCK_REALTIME, &curTime);
    start = curTime.tv_sec * 1000000000 + curTime.tv_nsec;

    switch (rendertarget) {
    case TARGET_CUDA:
        printf("Renderer: CUDA double precision\n");
        mandelbrotCUDA(rs);
        break;
    case TARGET_CUDASP:
        printf("Renderer: CUDA single precision\n");
        mandelbrotCUDAsp(rs);
        break;
    case TARGET_AVX:
        printf("Renderer: AVX\n");
        mandelbrotAVX(rs);
        break;
    case TARGET_GMP:
        printf("Renderer: GMP\n");
        mandelbrotGMP(rs);
        break;
    case TARGET_CPU:
        printf("Renderer: CPU\n");
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
    SDL_UnlockMutex(mutex);
}

void setupSDL() {
    SDL_Init(SDL_INIT_VIDEO);

    win = SDL_CreateWindow("sdltest", 0, 0, INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT,
                           SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    tex = SDL_CreateTexture(rend, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING,
                            rs.width, rs.height);
    mutex = SDL_CreateMutex();
}

// clang-format off
float vertices[] = {
    // positions         // texture coords
     1.0f,  1.0f, 0.0f,  1.0f, 0.0f, // top right
     1.0f, -1.0f, 0.0f,  1.0f, 1.0f, // bottom right
    -1.0f, -1.0f, 0.0f,  0.0f, 1.0f, // bottom left
    -1.0f,  1.0f, 0.0f,  0.0f, 0.0f  // top left
};
unsigned int indices[] = {
    0, 1, 3, // first triangle
    1, 2, 3  // second triangle
};
// clang-format on

void handleEvent(SDL_Event event) {
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
        case SDL_SCANCODE_MINUS:
            if (rs.zoom > 0.5)
                rs.zoom = rs.zoom / 1.5;
            renderWindow(rend, tex, rs);
            break;
        case SDL_SCANCODE_KP_DIVIDE:
            rs.zoom = 1.0;
            renderWindow(rend, tex, rs);
            break;
        case SDL_SCANCODE_KP_PLUS:
        case SDL_SCANCODE_EQUALS:
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
        case SDL_SCANCODE_G:
            SDL_Delay(1);
            SDL_GLContext glcontext = SDL_GL_CreateContext(win);

            GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
            GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

            FILE *fp;
            char buff[MAX_SHADER_SIZE];
            fp = fopen("shader.fs", "r");
            size_t count = fread(buff, 1, MAX_SHADER_SIZE, (FILE *)fp);
            buff[count] = '\0';
            fclose(fp);

            const char *shaderSource = buff;
            glShaderSource(fragmentShader, 1, &shaderSource, NULL);

            fp = fopen("shader.vs", "r");
            count = fread(buff, 1, MAX_SHADER_SIZE, (FILE *)fp);
            buff[count] = '\0';
            fclose(fp);
            glShaderSource(vertexShader, 1, &shaderSource, NULL);

            glCompileShader(vertexShader);
            glCompileShader(fragmentShader);

            GLuint shaderProgram = glCreateProgram();
            glAttachShader(shaderProgram, vertexShader);
            glAttachShader(shaderProgram, fragmentShader);
            glLinkProgram(shaderProgram);
            glDeleteShader(vertexShader);
            glDeleteShader(fragmentShader);

            GLuint VBO, VAO, EBO;
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);

            glBindVertexArray(VAO);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

            // position
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
            glEnableVertexAttribArray(0);
            // texture
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);

            GLuint texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            rs.width = 16384;
            rs.height = 16384;

            void *data;
            data = (unsigned *)malloc(rs.width * rs.height * 4);

            rs.outputBuffer = data;

            mandelbrotAVX(rs);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rs.width, rs.height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, data);
            glGenerateMipmap(GL_TEXTURE_2D);

            glBindTexture(GL_TEXTURE_2D, texture);

            glUseProgram(shaderProgram);
            glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized

            mat4 transform = GLM_MAT4_IDENTITY_INIT;

            for (int i = 0; i < 1000; i++) {

                glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

                // glm_scale(transform, (vec3) {1.01f, 1.01f, 1.01f});
                glm_rotate(transform, 0.003f, (vec3){0.0f, 0.0f, 1.0f});
                glm_rotate(transform, 0.005f, (vec3){0.0f, 1.0f, 0.0f});
                glm_rotate(transform, 0.007f, (vec3){1.0f, 0.0f, 0.0f});

                unsigned int transformLoc = glGetUniformLocation(shaderProgram, "transform");
                glUniformMatrix4fv(transformLoc, 1, GL_FALSE, (float *)transform);

                glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                SDL_GL_SwapWindow(win);
                SDL_Delay(16);
            }

            SDL_Delay(1000);

            glDeleteVertexArrays(1, &VAO);
            glDeleteBuffers(1, &VBO);
            glDeleteBuffers(1, &EBO);
            glDeleteProgram(shaderProgram);

            SDL_GL_DeleteContext(glcontext);
            free(data);
            break;
        case SDL_SCANCODE_W:
            rs.iterations = rs.iterations * 2;
            renderWindow(rend, tex, rs);
            break;
        case SDL_SCANCODE_S:
            if (rs.iterations > 1) {
                rs.iterations = rs.iterations / 2;
                renderWindow(rend, tex, rs);
            }
            break;
        case SDL_SCANCODE_1:
            rendertarget = 0;
            renderWindow(rend, tex, rs);
            SDL_Delay(100);
            break;
        case SDL_SCANCODE_2:
            rendertarget = 1;
            renderWindow(rend, tex, rs);
            SDL_Delay(100);
            break;
        case SDL_SCANCODE_3:
            rendertarget = 2;
            renderWindow(rend, tex, rs);
            SDL_Delay(100);
            break;
        case SDL_SCANCODE_4:
            rendertarget = 3;
            renderWindow(rend, tex, rs);
            SDL_Delay(100);
            break;
        case SDL_SCANCODE_5:
            rendertarget = 4;
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
        printf("\n");
        break;
    case SDL_WINDOWEVENT:
        // printf("%d\n", event.window.event);
        if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {

            mutexstatus = SDL_TryLockMutex(mutex);

            if (mutexstatus == 0) {
                SDL_DestroyTexture(tex);
                rs.width = event.window.data1;
                rs.height = event.window.data2;
                tex = SDL_CreateTexture(rend, SDL_PIXELFORMAT_RGBA8888,
                                        SDL_TEXTUREACCESS_STREAMING,
                                        rs.width, rs.height);
                renderWindow(rend, tex, rs);

                SDL_UnlockMutex(mutex);
            } else {
                fprintf(stderr, "Couldn't lock mutex in event loop\n");
            }
        }
        break;
    default:
        break;
        printf("%d\n", event.type);
    }
}

int main(int argc, char *argv[]) {

    rs.zoom = 1.0;
    rs.xoffset = 0;
    rs.yoffset = 0;
    rs.iterations = 50;
    rs.width = INITIAL_WINDOW_WIDTH;
    rs.height = INITIAL_WINDOW_HEIGHT;

    setupSDL();
    initCUDA(rs);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }
    fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        close_requested = 1;

        // arbitrarily chosen zoom point for benchmarking
        rs.xoffset = -1.484935782949454;
        rs.zoom = 16585998.481410;
        rs.iterations = 5000;

        // rs.xoffset = -1.478036884621246;
        // rs.zoom = 194.619507;
        // rs.iterations = 500;

        // rs.xoffset = -1.483321409799798;
        // rs.zoom = 16100687809804.728516;
        // rs.iterations = 25600;

        rendertarget = TARGET_CPU;
        renderWindow(rend, tex, rs);
        rendertarget = TARGET_AVX;
        renderWindow(rend, tex, rs);
        rendertarget = TARGET_CUDASP;
        renderWindow(rend, tex, rs);
        rendertarget = TARGET_CUDA;
    }

    renderWindow(rend, tex, rs);

    while (!close_requested) {
        SDL_Event event;
        SDL_WaitEvent(&event);
        handleEvent(event);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    freeCUDA();
    SDL_Quit();
}