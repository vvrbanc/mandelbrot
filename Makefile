CFLAGS := `sdl2-config --libs --cflags` --std=c11 -Wall -lSDL2_image -lm -march=native -mtune=native -pipe -Wall -fopenmp
# CFLAGS := $(CFLAGS) -ggdb3 -O0
CFLAGS := $(CFLAGS) -O3 -ffast-math -fomit-frame-pointer -fno-finite-math-only -flto

# CC := gcc
# HDRS :=
# SRCS := mandelmain.c
# OBJS := $(SRCS:.c=.o)
# EXEC := mandel

# $(EXEC): $(OBJS) $(HDRS) Makefile
# 	echo $(CC) -o $@ $(OBJS) $(CFLAGS)


all: mandel mandelbrot.run

rebuild: clean all

rebuildmr: clean mandelbrot.run

mandelbrot.run:
	# gcc -pipe -Wall -O3 -fomit-frame-pointer -fno-finite-math-only -march=native -fopenmp mandelbrot.gcc-6.c -o mandelbrot.run
	gcc -pipe -Wall -ggdb -O0 -march=native -fopenmp mandelbrot.gcc-6.c -o mandelbrot.run -DDEBUG

dlink.o: mandelcuda.o
	nvcc -O3 -dlink mandelcuda.o -o dlink.o -gencode arch=compute_61,code=sm_61

mandelcuda.o:
	nvcc -O3 -dc mandelcuda.cu -o mandelcuda.o -gencode arch=compute_61,code=sm_61

mandelcpu.o:
	gcc -c -o mandelcpu.o mandelcpu.c $(CFLAGS)

mandelmain.o:
	gcc -c -o mandelmain.o mandelmain.c $(CFLAGS)

mandel: dlink.o mandelmain.o mandelcpu.o
	gcc -o mandel mandelmain.o dlink.o mandelcuda.o mandelcpu.o $(CFLAGS) -L/opt/cuda/targets/x86_64-linux/lib -lm -lcudadevrt -lcudart

clean:
	rm -f mandelcuda.o mandelmain.o dlink.o mandel mandelcpu.o mandelbrot.run

.PHONY: all clean

