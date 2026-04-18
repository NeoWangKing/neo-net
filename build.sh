#!/bin/sh

set -xe

# export PKG_CONFIG_PATH=$HOME/opt/raylib/lib/pkgconfig/

CFLAGS="-O3 -Wall -Wextra `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lm -lglfw -ldl -lpthread -lX11 -lXrandr -lXinerama -lXi -lXcursor"

# gcc -Wall -Wextra -o xor xor.c -lm
# gcc $CFLAGS -o adder adder.c $LIBS
# gcc -Wall -Wextra -o dump_nn dump_nn.c -lm
# gcc $CFLAGS -o adder_gen adder_gen.c $LIBS
# gcc $CFLAGS -o xor_gen xor_gen.c $LIBS
gcc $CFLAGS -o gym gym.c $LIBS

# feh --geometry 1000x800 --auto-zoom --image-bg '#181818' *.png
