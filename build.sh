#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra"
LIBS="-lm"

# gcc $CFLAGS `pkg-config --cflags raylib` -o xor xor.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread -lX11 -lXrandr -lXinerama -lXi -lXcursor
# gcc $CFLAGS -o adder adder.c $LIBS

# gcc $CFLAGS -o adder_gen adder_gen.c $LIBS
# gcc $CFLAGS `pkg-config --cflags raylib` -o adder_gen adder_gen.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread -lX11 -lXrandr -lXinerama -lXi -lXcursor
# gcc $CFLAGS -o xor_gen xor_gen.c $LIBS
gcc $CFLAGS `pkg-config --cflags raylib` -o img2nn img2nn.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread -lX11 -lXrandr -lXinerama -lXi -lXcursor

# gcc $CFLAGS `pkg-config --cflags raylib` -o gym gym.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread -lX11 -lXrandr -lXinerama -lXi -lXcursor
