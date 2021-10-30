CFLAGS := --std=gnu17 -Wall -march=native -mtune=native -pipe -Wall -fopenmp
CFLAGS  := $(CFLAGS) `sdl2-config --cflags`
LDFLAGS := `sdl2-config --libs` -lSDL2_image -lm
LDFLAGS := $(LDFLAGS) -L/opt/cuda/targets/x86_64-linux/lib -lcudadevrt -lcudart -lGL -lGLEW -lgmp

# CFLAGS := $(CFLAGS) -ggdb3 -O0
CFLAGS := $(CFLAGS) -Ofast -ffast-math -fomit-frame-pointer -fno-finite-math-only -flto

CC = gcc

all: mandelbrot

new: clean all

dlink.o: mandelcuda.o
	nvcc -O3 -dlink mandelcuda.o -o dlink.o

mandelcuda.o: mandelcuda.cu
	nvcc -O3 -dc mandelcuda.cu -o mandelcuda.o

mandelcpu.o: mandelcpu.c
	$(CC) -c -o mandelcpu.o mandelcpu.c $(CFLAGS)

mandelmain.o: mandelmain.c
	$(CC) -c -o mandelmain.o mandelmain.c $(CFLAGS)

mandelbrot: dlink.o mandelmain.o mandelcpu.o
	$(CC) -o mandelbrot mandelmain.o dlink.o mandelcuda.o mandelcpu.o $(CFLAGS) $(LDFLAGS)

clean:
	rm -f mandelcuda.o mandelmain.o dlink.o mandelbrot mandelcpu.o

.PHONY: all clean

