CFLAGS := `sdl2-config --libs --cflags` --std=c11 -Wall -lSDL2_image -lm -ggdb3 -O0 -march=native -mtune=native -pipe -Wall
# CFLAGS := `sdl2-config --libs --cflags` --std=c11 -Wall -lSDL2_image -lm -O3 -march=native -mtune=native -pipe -Wall


CC := gcc
HDRS :=
SRCS := sdltest.c
OBJS := $(SRCS:.c=.o)
EXEC := sdltest


all: $(EXEC)


$(EXEC): $(OBJS) $(HDRS) Makefile
	$(CC) -o $@ $(OBJS) $(CFLAGS)

clean:
	rm -f $(EXEC) $(OBJS)

.PHONY: all clean



