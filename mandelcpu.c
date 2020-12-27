
#include "mandelmain.h"

void mandelbrotCPU(struct RenderSettings rs) {
    Uint32 *dst;

    double x1 = rs.x0 - 2.0 / rs.zoom + rs.xoffset;
    double x2 = rs.x0 + 2.0 / rs.zoom + rs.xoffset;
    double y1 = rs.y0 + 2.0 / rs.zoom + rs.yoffset;
    double y2 = rs.y0 - 2.0 / rs.zoom + rs.yoffset;

    double xpitch = (x2 - x1) / WINDOW_WIDTH;
    double ypitch = (y1 - y2) / WINDOW_HEIGHT;

    dst = (Uint32 *)((Uint8 *)rs.outputBuffer);

    int c = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < WINDOW_HEIGHT; y++)
    {
        double cReal, cImag, zReal, zImag, z2Real, z2Imag;
        Uint32 color;
        Uint32 colorbias;

        for (int x = 0; x < WINDOW_WIDTH; x++)
        {
            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)

            cReal = x1 + xpitch * x;
            cImag = y1 - ypitch * y;

            zReal = cReal;
            zImag = cImag;

            color = 0; // black as default for values that converge to 0

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 0; i < rs.iterations; i++)
            {
                z2Real = zReal * zReal - zImag * zImag + cReal;
                z2Imag = 2.0f * zReal * zImag + cImag;
                zReal = z2Real;
                zImag = z2Imag;

                if (zReal * zReal + zImag * zImag > 4.0f) {
                    colorbias = MIN(255, i * 510.0 / rs.iterations);
                    color = (0xFF000000 | (colorbias << 16) | (colorbias << 8) | colorbias);
                    #pragma omp atomic
                    c+=i;            
                    break;
                }
            }
            Uint32 *dst2 = (Uint32*)((Uint8 *)dst + x * 4 + y * WINDOW_WIDTH * 4);
            *dst2 = color;
        }
    }
    printf("total iterations: %d\n", c );
};


void mandelbrotAVX(struct RenderSettings rs) {

    Uint32 *dst;

    double x1 = rs.x0 - 2.0 / rs.zoom + rs.xoffset;
    double x2 = rs.x0 + 2.0 / rs.zoom + rs.xoffset;
    double y1 = rs.y0 + 2.0 / rs.zoom + rs.yoffset;
    double y2 = rs.y0 - 2.0 / rs.zoom + rs.yoffset;

    double xpitch = (x2 - x1) / WINDOW_WIDTH;
    double ypitch = (y1 - y2) / WINDOW_HEIGHT;

    __m256d vxpitch = _mm256_set1_pd(xpitch);
    __m256d vx1 = _mm256_set1_pd(x1);
    __m256d vOne = _mm256_set1_pd(1);
    __m256d vFour = _mm256_set1_pd(4);

    dst = (Uint32 *)((Uint8 *)rs.outputBuffer);

    int c = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < WINDOW_HEIGHT; y++)
    {
        __m256d vzrzi;
        __m256d vcImag = _mm256_set1_pd(y1 - ypitch * y);

        for (int x = 0; x < WINDOW_WIDTH; x+=4)
        {

            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)

            __m256d mx = _mm256_set_pd(x + 3, x + 2, x + 1, x);
            __m256d vcReal = _mm256_add_pd(_mm256_mul_pd(mx, vxpitch), vx1);

            __m256d vzReal = vcReal;
            __m256d vzImag = vcImag;

            __m256d vz2Real = vcReal;
            __m256d vz2Imag = vcImag;

            __m256d vIter = _mm256_set1_pd(0);

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 0; i < rs.iterations; i++)
            {

                vz2Real = _mm256_mul_pd(vzReal, vzReal);
                vz2Imag = _mm256_mul_pd(vzImag, vzImag);
                vzrzi = _mm256_mul_pd(vzReal, vzImag);

                vzReal = _mm256_add_pd(_mm256_sub_pd(vz2Real, vz2Imag), vcReal);
                vzImag = _mm256_add_pd(_mm256_add_pd(vzrzi, vzrzi), vcImag);

                __m256d mag2 = _mm256_add_pd(vz2Real, vz2Imag);
                __m256d mask = _mm256_cmp_pd(mag2, vFour, _CMP_LT_OS);
                vIter = _mm256_add_pd(_mm256_and_pd(mask, vOne), vIter);

                if (_mm256_testz_pd(mask, _mm256_set1_pd(-1))){
                    #pragma omp atomic
                    c+=i;            
                    break;
                }
            }

            // __m256d vColorbias = _mm256_set1_pd(510 / rs.iterations);
            // __m256d vColor = _mm256_mul_pd(vColorbias, mk);
            __m128i pixels = _mm256_cvtpd_epi32(vIter);

            unsigned int *dst2 = (unsigned int *)((Uint8 *)dst + x * 4 + y * WINDOW_WIDTH * 4);

            unsigned int x[4];
            x[0] = _mm_extract_epi32(pixels,0);
            x[1] = _mm_extract_epi32(pixels,1);
            x[2] = _mm_extract_epi32(pixels,2);
            x[3] = _mm_extract_epi32(pixels,3);

            int y;
            for (int j = 0; j < 4; j++) {
                y = x[j];
                if (y == rs.iterations){
                    y = 0xFF000000;
                }
                else {
                    // x = MIN(255, (int)255 * x / rs.iterations * 2);
                    y = MIN(255, y * 510.0 / rs.iterations);
                    y = (0xFF000000 | (y << 16) | (y << 8) | y);
                }
                
                dst2[j] = y;
            }
        }
    }
    printf("total iterations: %d\n", c );
};
