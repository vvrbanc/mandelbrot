#include "mandelcpu.h"
#include "mandelmain.h"
#include <math.h>

void mandelbrotCPU(struct RenderSettings rs) {

#define CPU_UNROLL 8
    // ^^ number of pixels iterated over in (sort of) parallel.
    //
    // each iteration needs previous one's zReal/zImag result,
    // the hot loop is dominated by FMA latency.
    //
    // pixels are mutually independent, so by processing more than 1 per iteration
    // we allow cpu's out-of-order scheduler to dispatch them in parallel
    // and saturate the cpu's FP ports. 
    // 
    // CPU_UNROLL = 8 empirically found to be goldilocks value for my cpu.
    //
    // AVX and NEON versions use the same unroll trick

    double x1 = rs.xoffset - 2.0 / rs.zoom * rs.width / rs.height;
    double x2 = rs.xoffset + 2.0 / rs.zoom * rs.width / rs.height;
    double y1 = rs.yoffset + 2.0 / rs.zoom;

    double pixel_pitch = (x2 - x1) / rs.width;
    double colorscale = 510.0 / rs.iterations;

#pragma omp parallel for schedule(dynamic) if (rs.multithreaded)
    for (int y = 0; y < rs.height; y++) {
        double cReal[CPU_UNROLL], cImag;
        cImag = y1 - pixel_pitch * y;

        for (int x = 0; x < rs.width; x += CPU_UNROLL) {
            double zReal[CPU_UNROLL], zImag[CPU_UNROLL];
            uint32_t iters[CPU_UNROLL];
            int live[CPU_UNROLL];
            int anyLive = 1;

            for (int k = 0; k < CPU_UNROLL; k++) {
                cReal[k] = x1 + pixel_pitch * (x + k);
                zReal[k] = cReal[k];
                zImag[k] = cImag;
                iters[k] = 0;
                live[k] = 1;
            }

            for (uint i = 0; i < rs.iterations && anyLive; i++) {
                anyLive = 0;
                for (int k = 0; k < CPU_UNROLL; k++) {
                    double mag2 = fma(zReal[k], zReal[k], zImag[k] * zImag[k]); // |z|^2 = zReal^2 + zImag^2
                    double tmpval = fma(-zImag[k], zImag[k], cReal[k]);         // cReal - zImag^2: the part of the next real value that doesn't need zReal^2 yet
                    zImag[k] = fma(zReal[k] + zReal[k], zImag[k], cImag);       // z' = z^2 + c, imaginary part: 2 * zReal * zImag + cImag
                    zReal[k] = fma(zReal[k], zReal[k], tmpval);                 // z' = z^2 + c, real part: zReal^2 - zImag^2 + cReal

                    int inside = live[k] & (mag2 <= 4.0);
                    iters[k] += inside;
                    live[k] = inside;
                    anyLive |= inside;
                }
            }
            for (int k = 0; k < CPU_UNROLL; k++) {
                uint32_t color = 0x000000FF;
                if (iters[k] != rs.iterations) {
                    uint32_t colorbias = MIN(255, iters[k] * colorscale);
                    color = (0x000000FF | (colorbias << 24) | (colorbias << 16) | colorbias << 8);
                }
                rs.outputBuffer[x + k + y * rs.width] = color;
            }
        }
    }
}
#ifdef __AVX__

#define AVX_UNROLL 2
#define AVX_CHECK_INTERVAL 8
// ^^ iterations between all-lanes-escaped checks (compare+branch is off the hot path)

void mandelbrotAVX(struct RenderSettings rs) {

    double x1 = rs.xoffset - 2.0 / rs.zoom * rs.width / rs.height;
    double x2 = rs.xoffset + 2.0 / rs.zoom * rs.width / rs.height;
    double y1 = rs.yoffset + 2.0 / rs.zoom;

    double pixel_pitch = (x2 - x1) / rs.width;
    double colorscale = 510.0 / rs.iterations;

    __m256d vxpitch = _mm256_set1_pd(pixel_pitch);
    __m256d vx1 = _mm256_set1_pd(x1);
    __m256d vFour = _mm256_set1_pd(4);

#pragma omp parallel for schedule(dynamic) if (rs.multithreaded)
    for (int y = 0; y < rs.height; y++) {
        __m256d vcImag = _mm256_set1_pd(y1 - pixel_pitch * y);

        for (int x = 0; x < rs.width; x += 4 * AVX_UNROLL) {
            __m256d vcReal[AVX_UNROLL], vzReal[AVX_UNROLL], vzImag[AVX_UNROLL];
            __m256i vIter[AVX_UNROLL];

            for (int k = 0; k < AVX_UNROLL; k++) {
                int xk = x + 4 * k;

                __m256d mx = _mm256_set_pd(xk + 3, xk + 2, xk + 1, xk);
                vcReal[k] = _mm256_fmadd_pd(mx, vxpitch, vx1);
                vzReal[k] = vcReal[k];
                vzImag[k] = vcImag;
                vIter[k] = _mm256_setzero_si256();
            }

            for (uint i = 0; i < rs.iterations; i += AVX_CHECK_INTERVAL) {
                uint n = MIN(AVX_CHECK_INTERVAL, rs.iterations - i); // check for escape only once per AVX_CHECK_INTERVAL
                __m256d vMask[AVX_UNROLL];

                for (uint j = 0; j < n; j++) {
                    for (int k = 0; k < AVX_UNROLL; k++) {
                        __m256d mag2 = _mm256_fmadd_pd(vzReal[k], vzReal[k], _mm256_mul_pd(vzImag[k], vzImag[k]));
                        __m256d tmpval = _mm256_fnmadd_pd(vzImag[k], vzImag[k], vcReal[k]);
                        vzImag[k] = _mm256_fmadd_pd(_mm256_add_pd(vzReal[k], vzReal[k]), vzImag[k], vcImag);
                        vzReal[k] = _mm256_fmadd_pd(vzReal[k], vzReal[k], tmpval);
                        vMask[k] = _mm256_cmp_pd(mag2, vFour, _CMP_LT_OQ);
                        // mask lanes are all-ones (-1) when inside, so subtracting counts the iteration
                        vIter[k] = _mm256_sub_epi64(vIter[k], _mm256_castpd_si256(vMask[k]));
                    }
                }

                __m256d anyInside = vMask[0];
                for (int k = 1; k < AVX_UNROLL; k++) {
                    anyInside = _mm256_or_pd(anyInside, vMask[k]);
                }
                if (_mm256_testz_pd(anyInside, anyInside)) {
                    break;
                }
            }

            for (int k = 0; k < AVX_UNROLL; k++) {
                uint64_t iters[4];
                _mm256_storeu_si256((__m256i *)iters, vIter[k]);

                for (int ii = 0; ii < 4; ii++) {
                    uint32_t color, colorbias;
                    if (iters[ii] == rs.iterations) {
                        color = 0x000000FF;
                    } else {
                        colorbias = MIN(255, iters[ii] * colorscale);
                        color = (0x000000FF | (colorbias << 24) | (colorbias << 16) | colorbias << 8);
                    }
                    rs.outputBuffer[x + 4 * k + y * rs.width + ii] = color;
                }
            }
        }
    }
}

#endif

#ifdef __aarch64__

#define NEON_CHECK_INTERVAL 8

void mandelbrotNEON(struct RenderSettings rs) {

    double x1 = rs.xoffset - 2.0 / rs.zoom * rs.width / rs.height;
    double x2 = rs.xoffset + 2.0 / rs.zoom * rs.width / rs.height;
    double y1 = rs.yoffset + 2.0 / rs.zoom;

    double pixel_pitch = (x2 - x1) / rs.width;

    float64x2_t vxpitch = vdupq_n_f64(pixel_pitch);
    float64x2_t vx1 = vdupq_n_f64(x1);
    float64x2_t vFour = vdupq_n_f64(4.0);

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < rs.height; y++) {
        double cImag = y1 - pixel_pitch * y;
        float64x2_t vcImag = vdupq_n_f64(cImag);

        for (int x = 0; x < rs.width; x += 2) {
            double xs[2] = {x, x + 1};
            float64x2_t vcReal = vfmaq_f64(vx1, vld1q_f64(xs), vxpitch);

            float64x2_t vzReal = vcReal;
            float64x2_t vzImag = vcImag;
            uint64x2_t vIter = vdupq_n_u64(0);

            for (uint i = 0; i < rs.iterations; i += NEON_CHECK_INTERVAL) {
                uint n = MIN(NEON_CHECK_INTERVAL, rs.iterations - i);
                uint64x2_t mask;
                for (uint j = 0; j < n; j++) {
                    float64x2_t mag2 = vfmaq_f64(vmulq_f64(vzImag, vzImag), vzReal, vzReal);
                    float64x2_t tmpval = vfmsq_f64(vcReal, vzImag, vzImag);
                    vzImag = vfmaq_f64(vcImag, vaddq_f64(vzReal, vzReal), vzImag);
                    vzReal = vfmaq_f64(tmpval, vzReal, vzReal);
                    mask = vcltq_f64(mag2, vFour);
                    vIter = vsubq_u64(vIter, mask);
                }

                if ((vgetq_lane_u64(mask, 0) | vgetq_lane_u64(mask, 1)) == 0) {
                    break;
                }
            }

            uint64_t iters[2] = {vgetq_lane_u64(vIter, 0), vgetq_lane_u64(vIter, 1)};

            for (int ii = 0; ii < 2; ii++) {
                uint32_t color, colorbias;
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
}
#endif

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

#pragma omp parallel for schedule(dynamic) if (rs.multithreaded)
    for (int y = 0; y < rs.height; y++) {
        mpf_t gcReal, gcImag, gzReal, gzImag, gz2Real, gz2Imag, gzrzi, gzTmp;
        mpf_inits(gcReal, gcImag, gzReal, gzImag, gz2Real, gz2Imag, gzrzi, gzTmp, NULL);
        uint32_t color;
        uint32_t colorbias;

        for (int x = 0; x < rs.width; x++) {
            // map screen coords to (0,0) -> (-2,2) through (WW,WH) -> (2, -2)

            mpf_mul_ui(gcReal, gpixel_pitch, x);
            mpf_add(gcReal, gx1, gcReal);

            mpf_mul_ui(gcImag, gpixel_pitch, y);
            mpf_sub(gcImag, gy1, gcImag);

            mpf_set(gzReal, gcReal);
            mpf_set(gzImag, gcImag);

            color = 0; // black as default for values that converge to 0

            for (uint i = 0; i < rs.iterations; i++) {
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
}
