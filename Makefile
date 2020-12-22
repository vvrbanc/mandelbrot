CFLAGS := `sdl2-config --libs --cflags` --std=c11 -Wall -lSDL2_image -lm -march=native -mtune=native -pipe -Wall
# CFLAGS := $(CFLAGS) -ggdb3 -O0
CFLAGS := $(CFLAGS) -O3

# CC := gcc
# HDRS :=
# SRCS := mandelmain.c
# OBJS := $(SRCS:.c=.o)
# EXEC := mandel

# $(EXEC): $(OBJS) $(HDRS) Makefile
# 	echo $(CC) -o $@ $(OBJS) $(CFLAGS)


all: mandel

rebuild: clean mandel

dlink.o: mandelcuda.o
	nvcc -O3 -dlink mandelcuda.o -o dlink.o -gencode arch=compute_61,code=sm_61 -dlink

mandelcuda.o:
	nvcc -O3 -dc mandelcuda.cu -o mandelcuda.o -gencode arch=compute_61,code=sm_61

mandelcpu.o:
	gcc $(CFLAGS) -c -o mandelcpu.o mandelcpu.c

mandelmain.o:
	gcc $(CFLAGS) -c -o mandelmain.o mandelmain.c

mandel: dlink.o mandelmain.o mandelcpu.o
	gcc $(CFLAGS) -o mandel mandelmain.o dlink.o mandelcuda.o mandelcpu.o -L/opt/cuda/targets/x86_64-linux/lib -lm -lcudadevrt -lcudart

clean:
	rm -f mandelcuda.o mandelmain.o dlink.o mandel mandelcpu.o

.PHONY: all clean

