CFLAGS := `sdl2-config --libs --cflags` --std=c11 -Wall -lSDL2_image -lm -march=native -mtune=native -pipe -Wall -fopenmp
# CFLAGS := $(CFLAGS) -ggdb3 -O0
CFLAGS := $(CFLAGS) -O3 -ffast-math -fomit-frame-pointer -fno-finite-math-only -flto


all: mandel

rebuild: clean all

dlink.o: mandelcuda.o
	nvcc -O3 -dlink mandelcuda.o -o dlink.o

mandelcuda.o:
	nvcc -O3 -dc mandelcuda.cu -o mandelcuda.o

mandelcpu.o:
	gcc -c -o mandelcpu.o mandelcpu.c $(CFLAGS)

mandelmain.o:
	gcc -c -o mandelmain.o mandelmain.c $(CFLAGS)

mandel: dlink.o mandelmain.o mandelcpu.o
	gcc -o mandel mandelmain.o dlink.o mandelcuda.o mandelcpu.o $(CFLAGS) -L/opt/cuda/targets/x86_64-linux/lib -lm -lcudadevrt -lcudart

clean:
	rm -f mandelcuda.o mandelmain.o dlink.o mandel mandelcpu.o

.PHONY: all clean

