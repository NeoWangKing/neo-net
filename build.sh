#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra"
LIBS="-lm"


gcc $CFLAGS `pkg-config --cflags raylib` -o xor_gen xor_gen.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread -lX11 -lXrandr -lXinerama -lXi -lXcursor
gcc $CFLAGS `pkg-config --cflags raylib` -o adder_gen adder_gen.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread -lX11 -lXrandr -lXinerama -lXi -lXcursor
gcc $CFLAGS `pkg-config --cflags raylib` -o img2nn img2nn.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread -lX11 -lXrandr -lXinerama -lXi -lXcursor

