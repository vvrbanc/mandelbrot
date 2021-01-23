
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

            // convert 4x double vector (256) to 4x int32 (128) and copy to ram as uint32[4]
            Uint32 iters[4];
            _mm_store_si128((__m128i *)iters, _mm256_cvtpd_epi32(vIter));

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

void mandelbrotGMP(struct RenderSettings rs) {

    mpf_set_default_prec(96);

    mpf_t gx1, gx2, gy1, gpixel_pitch, gTmp;
    mpf_inits(gx1, gx2, gy1, gpixel_pitch, gTmp, NULL);

    mpf_set_d(gx1, 2.0f * rs.width / rs.height);
    mpf_set_d(gTmp, rs.zoom);
    mpf_div(gx1, gx1, gTmp);

    mpf_set_d(gy1, 2.0f);
    mpf_div(gy1, gy1, gTmp);

    mpf_set_d(gTmp, rs.xoffset);
    mpf_add(gx2, gTmp, gx1);
    mpf_sub(gx1, gTmp, gx1);

    mpf_set_d(gTmp, rs.yoffset);
    mpf_add(gy1, gTmp, gy1);

    mpf_set_d(gTmp, rs.width);
    mpf_sub(gpixel_pitch, gx2, gx1);
    mpf_div(gpixel_pitch, gpixel_pitch, gTmp);

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < rs.height; y++) {
        mpf_t gcReal, gcImag, gzReal, gzImag, gz2Real, gz2Imag, gzrzi, gzTmp;
        mpf_inits(gcReal, gcImag, gzReal, gzImag, gz2Real, gz2Imag, gzrzi, gzTmp, NULL);
        Uint32 color;
        Uint32 colorbias;

        for (int x = 0; x < rs.width; x++) {
            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)

            mpf_mul_ui(gcReal, gpixel_pitch, x);
            mpf_add(gcReal, gx1, gcReal);

            mpf_mul_ui(gcImag, gpixel_pitch, y);
            mpf_sub(gcImag, gy1, gcImag);

            mpf_set(gzReal, gcReal);
            mpf_set(gzImag, gcImag);

            color = 0; // black as default for values that converge to 0

            // Mandelbrot calc for current (x,y) pixel
            for (int i = 0; i < rs.iterations; i++) {
                mpf_mul(gz2Real, gzReal, gzReal);
                mpf_mul(gz2Imag, gzImag, gzImag);
                mpf_add(gzTmp, gz2Real, gz2Imag);

                if (mpf_cmp_ui(gzTmp, 4) > 0) {
                    colorbias = MIN(255, i * 510.0 / rs.iterations);
                    color = (0x000000FF | (colorbias << 24) | (colorbias << 16) | colorbias << 8);
                    break;
                }
                mpf_mul(gzrzi, gzReal, gzImag);

                mpf_add(gzReal, gcReal, gz2Real);
                mpf_sub(gzReal, gzReal, gz2Imag);

                mpf_add(gzImag, gzrzi, gzrzi);
                mpf_add(gzImag, gzImag, gcImag);
            }
            rs.outputBuffer[x + y * rs.width] = color;
        }
        mpf_clears(gcReal, gcImag, gzReal, gzImag, gz2Real, gz2Imag, gzrzi, gzTmp, NULL);
    }
};
