#!/bin/sh

set -xe

export PKG_CONFIG_PATH=$HOME/opt/raylib/lib/pkgconfig/:$PKG_CONFIG_PATH

CFLAGS="-O3 -Wall -Wextra"
LIBS="-lm $(pkg-config --libs raylib) -lX11 -lXrandr -lXi -lXcursor -lXinerama -lpthread -ldl"

gcc $CFLAGS `pkg-config --cflags raylib` -o xor_gen xor_gen.c $LIBS

gcc $CFLAGS `pkg-config --cflags raylib` -o adder_gen adder_gen.c $LIBS

gcc $CFLAGS `pkg-config --cflags raylib` -o img2nn img2nn.c $LIBS

# gcc -O3 -Wall -Wextra \
#     -o adder_gen adder_gen.c \
#     -I/opt/homebrew/include \
#     -L/opt/homebrew/lib \
#     -lraylib -lm \
#     -framework OpenGL \
#     -framework Cocoa \
#     -framework IOKit \
#     -framework CoreVideo
# gcc -O3 -Wall -Wextra \
#     -o xor_gen xor_gen.c \
#     -I/opt/homebrew/include \
#     -L/opt/homebrew/lib \
#     -lraylib -lm \
#     -framework OpenGL \
#     -framework Cocoa \
#     -framework IOKit \
#     -framework CoreVideo
# gcc -O3 -Wall -Wextra \
#     -o img2nn img2nn.c \
#     -I/opt/homebrew/include \
#     -L/opt/homebrew/lib \
#     -lraylib -lm \
#     -framework OpenGL \
#     -framework Cocoa \
#     -framework IOKit \
#     -framework CoreVideo
# ./img2nn ./mnist/train/8/train_image_10001.png ./mnist/train/6/train_image_10017.png
