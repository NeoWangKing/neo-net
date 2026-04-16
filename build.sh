#!/bin/sh

set -xe

# gcc -Wall -Wextra -o twice twice.c -lm
# gcc -Wall -Wextra -o main main.c -lm
# gcc -Wall -Wextra -o xor xor.c -lm
gcc -Wall -Wextra -o build/adder adder.c -lm

