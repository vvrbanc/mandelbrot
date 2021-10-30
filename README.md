# mandelbrot

A naive mandelbrot renderer written in C.

The purpose was to explore computationally-intensive operations in C. Also for me to learn C :)

Supports 5 ways of computing the set:
* regular C implementation
* AVX
* CUDA single precision
* CUDA double precision
* GMP (gnu mp bignum library)

CPU renderers all use OpenMP to spread workload across all cores.

## building

on ubuntu 20.10:
apt install gcc g++ make libsdl2-dev libsdl2-image-dev libcglm-dev libglew-dev libgmp-dev nvidia-cuda-toolkit

"make" to build

## key bindings

* 1-5: select renderer type
* W,S: increase/decrease number of iterations
* cursor keys: pan
* +,-: zoom
* ESC: quit

## remarks

CUDA double-precision version is relatively slow on most nvidia cards. A few Tesla-line cards have double-precision math that is only 2x slower than single-precision. On most other cards it is 20-50x slower.

GMP is *very* slow but does not lose precision at high zoom levels.