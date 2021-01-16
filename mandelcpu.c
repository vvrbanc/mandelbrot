
#include "mandelmain.h"

void mandelbrotCPU(struct RenderSettings rs) {

    double x1 = rs.xoffset - 2.0 / rs.zoom * rs.width / rs.height;
    double x2 = rs.xoffset + 2.0 / rs.zoom * rs.width / rs.height;
    double y1 = rs.yoffset + 2.0 / rs.zoom;

    double pixel_pitch = (x2 - x1) / rs.width;

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < rs.height; y++) {
        double cReal, cImag, zReal, zImag, z2Real, z2Imag, zrzi;
        Uint32 color;
        Uint32 colorbias;

        for (int x = 0; x < rs.width; x++) {
            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)

            cReal = x1 + pixel_pitch * x;
            cImag = y1 - pixel_pitch * y;

            zReal = cReal;
            zImag = cImag;

            color = 0; // black as default for values that converge to 0

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 0; i < rs.iterations; i++) {
                z2Real = zReal * zReal;
                z2Imag = zImag * zImag;
                zrzi = zReal * zImag;

                zReal = cReal + z2Real - z2Imag;
                zImag = zrzi + zrzi + cImag;

                if (z2Real + z2Imag > 4.0f) {
                    colorbias = MIN(255, i * 510.0 / rs.iterations);
                    color = (0x000000FF | (colorbias << 24) | (colorbias << 16) | colorbias << 8);
                    break;
                }
            }
            rs.outputBuffer[x + y * rs.width] = color;
        }
    }
};

void mandelbrotAVX(struct RenderSettings rs) {

    double x1 = rs.xoffset - 2.0 / rs.zoom * rs.width / rs.height;
    double x2 = rs.xoffset + 2.0 / rs.zoom * rs.width / rs.height;
    double y1 = rs.yoffset + 2.0 / rs.zoom;

    double pixel_pitch = (x2 - x1) / rs.width;

    __m256d vxpitch = _mm256_set1_pd(pixel_pitch);
    __m256d vx1 = _mm256_set1_pd(x1);
    __m256d vOne = _mm256_set1_pd(1);
    __m256d vFour = _mm256_set1_pd(4);

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < rs.height; y++) {
        __m256d vzrzi;
        __m256d vcImag = _mm256_set1_pd(y1 - pixel_pitch * y);

        for (int x = 0; x < rs.width - (rs.width % 4); x += 4) {

            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)

            __m256d mx = _mm256_set_pd(x + 3, x + 2, x + 1, x);
            __m256d vcReal = _mm256_add_pd(_mm256_mul_pd(mx, vxpitch), vx1);

            __m256d vzReal = vcReal;
            __m256d vzImag = vcImag;

            __m256d vz2Real = vcReal;
            __m256d vz2Imag = vcImag;

            __m256d vIter = _mm256_set1_pd(0);

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 0; i < rs.iterations; i++) {

                vz2Real = _mm256_mul_pd(vzReal, vzReal);
                vz2Imag = _mm256_mul_pd(vzImag, vzImag);
                vzrzi = _mm256_mul_pd(vzReal, vzImag);

                vzReal = _mm256_add_pd(_mm256_sub_pd(vz2Real, vz2Imag), vcReal);
                vzImag = _mm256_add_pd(_mm256_add_pd(vzrzi, vzrzi), vcImag);

                __m256d mag2 = _mm256_add_pd(vz2Real, vz2Imag);
                __m256d mask = _mm256_cmp_pd(mag2, vFour, _CMP_LT_OQ);
                vIter = _mm256_add_pd(_mm256_and_pd(mask, vOne), vIter);

                if (_mm256_testz_pd(mask, _mm256_set1_pd(-1))) {
                    break;
                }
            }

            // convert 4x double vector (256) to 4x uint32_t (128) and copy to ram as uint32 [4]
            Uint32 iters[4];
            _mm_store_si128((__m128i *) iters, _mm256_cvtpd_epi32(vIter));

            Uint32 color, colorbias;

            // calculate color for the 4 dumped pixels
            for (int ii = 0; ii < 4; ii++) {
                if (iters[ii] == rs.iterations) {
                    color = 0x000000FF;
                } else {
                    colorbias = MIN(255, iters[ii] * 510.0 / rs.iterations);
                    color = (0x000000FF | (colorbias << 24) | (colorbias << 16) | colorbias << 8);
                }
                rs.outputBuffer[x + y * rs.width + ii] = color;
            }
        }
    }
};